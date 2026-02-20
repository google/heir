#ifndef LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H
#define LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H

#include <cstddef>

#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
class ILPBootstrapPlacementAnalysis {
 public:
  ILPBootstrapPlacementAnalysis(Operation* op, DataFlowSolver* solver,
                                int bootstrapWaterline)
      : opToRunOn(op), solver(solver), bootstrapWaterline(bootstrapWaterline) {}
  ~ILPBootstrapPlacementAnalysis() = default;

  LogicalResult solve();

  // Return true if a bootstrap op should be inserted after the given
  // operation, according to the solution to the optimization problem.
  bool shouldInsertBootstrap(Operation* op) const {
    return solution.lookup(op);
  }

  // Return true if a bootstrap should be inserted after the i-th op (in body
  // order) that produces secret results. Use this after IR changes so
  // pointer-based lookup is not relied on. Index 0 is the first such op.
  bool shouldInsertBootstrapForOpIndex(size_t opIndex) const {
    return opIndex < bootstrapByOpIndex.size() && bootstrapByOpIndex[opIndex];
  }

  // Return a copy of the bootstrap-by-op-index decisions. Use this when the
  // IR will be modified after solve() so the pass does not rely on the
  // analysis object after pointer-invalidating edits.
  llvm::SmallVector<bool, 32> getBootstrapByOpIndexCopy() const {
    return llvm::SmallVector<bool, 32>(bootstrapByOpIndex.begin(),
                                       bootstrapByOpIndex.end());
  }

  // Number of bootstraps in the solution (for consistency checks).
  size_t getSolutionBootstrapCount() const {
    size_t n = 0;
    for (const auto& [op, insert] : solution)
      if (insert) ++n;
    return n;
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

 private:
  Operation* opToRunOn;
  DataFlowSolver* solver;
  int bootstrapWaterline;
  llvm::DenseMap<Operation*, bool> solution;
  llvm::DenseMap<Value, int> solutionLevelBeforeBootstrap;
  llvm::DenseMap<Value, int> solutionLevelAfterBootstrap;
  // Ops with secret results in body order; bootstrapByOpIndex[i] = solution for
  // i-th such op.
  llvm::SmallVector<Operation*, 32> orderedOps;
  llvm::SmallVector<bool, 32> bootstrapByOpIndex;
};
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ILP_BOOTSTRAP_PLACEMENT_ANALYSIS_H
