#ifndef LIB_ANALYSIS_UTILS_H_
#define LIB_ANALYSIS_UTILS_H_

#include <functional>

#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

// A generalized version of visitExternalCall that joins all arguments to the
// func.call op, and then propagates this joined value to all results. This is
// useful for debug functions where there is a single operand that is not
// changed by the external call, but can also be useful for some analyses like
// secretness, where the result of the call is secret if any operand is secret.
template <typename StateT, typename LatticeT>
void visitExternalCall(CallOpInterface call,
                       ArrayRef<const LatticeT*> argumentLattices,
                       ArrayRef<LatticeT*> resultLattices,
                       const std::function<void(AnalysisState*, ChangeResult)>&
                           propagateIfChanged) {
  StateT resultState = StateT();

  for (const LatticeT* operand : argumentLattices) {
    const StateT operandState = operand->getValue();
    resultState = StateT::join(resultState, operandState);
  }

  for (LatticeT* result : resultLattices) {
    propagateIfChanged(result, result->join(resultState));
  }
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_UTILS_H_
