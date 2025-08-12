#ifndef LIB_ANALYSIS_OPTIMIZE_RELINEARIZATIONANALYSIS_H
#define LIB_ANALYSIS_OPTIMIZE_RELINEARIZATIONANALYSIS_H

#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
class OptimizeRelinearizationAnalysis {
 public:
  OptimizeRelinearizationAnalysis(Operation* op, DataFlowSolver* solver,
                                  bool useLocBasedVariableNames,
                                  bool allowMixedDegreeOperands)
      : opToRunOn(op),
        solver(solver),
        useLocBasedVariableNames(useLocBasedVariableNames),
        allowMixedDegreeOperands(allowMixedDegreeOperands) {}
  ~OptimizeRelinearizationAnalysis() = default;

  LogicalResult solve();

  // Return true if a relin op should be inserted after the given
  // operation, according to the solution to the optimization problem.
  bool shouldInsertRelin(Operation* op) const { return solution.lookup(op); }

  // Return the key basis degree at the given SSA value, as determined by the
  // solution to the optimization problem. When the input value is the result
  // of an op, and the model solution suggests a relinearization should be
  // inserted after that op, this function returns the pre-relinearization
  // degree.
  //
  // We don't need an "after relin" version because after relin the key basis
  // is (1, s), which is degree 1.
  int keyBasisDegreeBeforeRelin(Value value) const {
    return solutionKeyBasisDegreeBeforeRelin.lookup(value);
  }

 private:
  Operation* opToRunOn;
  DataFlowSolver* solver;
  bool useLocBasedVariableNames;
  bool allowMixedDegreeOperands;
  llvm::DenseMap<Operation*, bool> solution;
  llvm::DenseMap<Value, int> solutionKeyBasisDegreeBeforeRelin;
};
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_OPTIMIZE_RELINEARIZATIONANALYSIS_H
