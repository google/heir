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
class ILPBootstrapPlacementAnalysis {
 public:
  enum class ScaleMode { kCKKS, kLevelOnly };

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
                                int bootstrapWaterline, int scaleWaterline,
                                int scaleFactorBits,
                                int bootstrapLevelLowerBound, int bootstrapCost,
                                int rescaleCost, ScaleMode scaleMode)
      : opToRunOn(op),
        solver(solver),
        bootstrapWaterline(bootstrapWaterline),
        scaleWaterline(scaleWaterline),
        scaleFactorBits(scaleFactorBits),
        bootstrapLevelLowerBound(bootstrapLevelLowerBound),
        bootstrapCost(bootstrapCost),
        rescaleCost(rescaleCost),
        scaleMode(scaleMode) {}
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
  int bootstrapWaterline;
  int scaleWaterline;
  int scaleFactorBits;
  int bootstrapLevelLowerBound;
  int bootstrapCost;
  int rescaleCost;
  ScaleMode scaleMode;
  llvm::DenseMap<Operation*, bool> solution;
  llvm::DenseMap<Value, int> solutionLevelBeforeBootstrap;
  llvm::DenseMap<Value, int> solutionLevelAfterBootstrap;
  llvm::SmallVector<NodeManagement, 32> nodeManagement;
  llvm::SmallVector<EdgeManagement, 32> edgeManagement;
};
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H
