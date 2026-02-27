#ifndef LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H
#define LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H

#include <cstddef>

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
  ILPBootstrapPlacementAnalysis(Operation* op, DataFlowSolver* solver,
                                int bootstrapWaterline)
      : opToRunOn(op), solver(solver), bootstrapWaterline(bootstrapWaterline) {}
  ~ILPBootstrapPlacementAnalysis() = default;

  LogicalResult solve();

  // Return the set of SSA values that the ILP decided should have a bootstrap
  // inserted after their definition. Used by the pass to insert bootstraps
  // without iterating by op index; Values remain valid across modreduce/
  // relinearize insertion.
  llvm::SmallVector<Value, 32> getValuesToBootstrap() const;

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

  // Debug: print the entire solution (bootstrap decisions and levels) to os.
  void printSolution(llvm::raw_ostream& os) const;

 private:
  Operation* opToRunOn;
  DataFlowSolver* solver;
  int bootstrapWaterline;
  llvm::DenseMap<Operation*, bool> solution;
  llvm::DenseMap<Value, int> solutionLevelBeforeBootstrap;
  llvm::DenseMap<Value, int> solutionLevelAfterBootstrap;
};
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H
