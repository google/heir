#ifndef LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H
#define LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H

#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace heir {

// A latency model of the form cost(level) = slope * level + intercept,
// fitted from a per-level latency table.
struct LinearCost {
  double slope = 0.0;
  double intercept = 0.0;
};

// Costs used by the ILP objective. Bootstrap and rescale management decisions
// are charged constant costs. When hasLevelCosts is set, each tracked op is
// additionally charged a level-dependent latency at its input level,
// distinguishing ciphertext-ciphertext (CtCt) from ciphertext-plaintext (CtPt)
// operands.
struct OpCostModel {
  double bootstrapCost = 0.0;
  double rescaleCost = 0.0;
  bool hasLevelCosts = false;
  LinearCost addCtCt;
  LinearCost addCtPt;
  LinearCost mulCtCt;
  LinearCost mulCtPt;
  LinearCost rotate;
  LinearCost negate;
};

class ILPBootstrapPlacementAnalysis {
 public:
  enum class ScaleMode { kCKKS, kLevelOnly };

  struct Options {
    int bootstrapWaterline = 0;
    int scaleWaterline = 0;
    int scaleFactorBits = 0;
    int bootstrapLevelLowerBound = 0;
    // Group structurally equivalent ops so they share ILP variables (Orbit's
    // compression). Grouped and ungrouped models have
    // the same optimum restricted to symmetric solutions.
    bool compress = true;
    // Minimum number of ops per SISO partition (Orbit's delta). Partitions
    // are solved independently under enumerated boundary states and stitched
    // by dynamic programming.
    int partitionMinSize = 100;
    OpCostModel costModel;
    ScaleMode scaleMode = ScaleMode::kCKKS;
  };

  struct NodeManagement {
    Value value;
    int inputLevel;
    int inputScale;
    int outputLevel;
    int outputScale;
    bool useBootstrap;
  };

  struct EdgeManagement {
    Operation* op;
    unsigned operandNumber;
    int inputLevel;
    int inputScale;
    int outputLevel;
    int outputScale;
  };

  ILPBootstrapPlacementAnalysis(Operation* op, DataFlowSolver* solver,
                                const Options& options)
      : opToRunOn(op), solver(solver), options(options) {}
  ~ILPBootstrapPlacementAnalysis() = default;

  LogicalResult solve();

  // Return per-result management transitions chosen by the ILP.
  llvm::SmallVector<NodeManagement, 32> getNodeManagement() const {
    return nodeManagement;
  }

  // Return per-use management transitions chosen by the ILP.
  llvm::SmallVector<EdgeManagement, 32> getEdgeManagement() const {
    return edgeManagement;
  }

  // Return the level at the given SSA value, as determined by the
  // solution to the optimization problem. When the input value is the result
  // of an op, and the model solution suggests a bootstrap should be
  // inserted after that op, this function returns the pre-bootstrap level.
  //
  // We don't need an "after bootstrap" version because after bootstrap the
  // level is reset to the maximum level (or a target level if specified).
  int levelBeforeBootstrap(Value value) const {
    return solutionLevelBeforeBootstrap.lookup(value);
  }

  // Debug: print the entire solution (bootstrap/rescale decisions and levels)
  // to os.
  void printSolution(llvm::raw_ostream& os) const;

 private:
  Operation* opToRunOn;
  DataFlowSolver* solver;
  Options options;
  llvm::DenseMap<Operation*, bool> solution;
  llvm::DenseMap<Value, int> solutionLevelBeforeBootstrap;
  llvm::DenseMap<Value, int> solutionLevelAfterBootstrap;
  llvm::SmallVector<NodeManagement, 32> nodeManagement;
  llvm::SmallVector<EdgeManagement, 32> edgeManagement;
};
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H
