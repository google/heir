#include "lib/Analysis/Cpp/ConstQualifierAnalysis.h"

#include "lib/Utils/Tablegen/InPlaceOpInterface.h"
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"    // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

bool isMutated(Value value) {
  return llvm::any_of(value.getUsers(), [&](Operation* user) {
    bool mutatesOperand = hasEffect<MemoryEffects::Write>(user, value);
    auto inPlaceOp = dyn_cast<InPlaceOpInterface>(user);
    bool isTargetOfInPlaceOp =
        inPlaceOp &&
        user->getOperand(inPlaceOp.getInPlaceOperandIndex()) == value;
    return mutatesOperand || isTargetOfInPlaceOp;
  });
}

ConstQualifierAnalysis::ConstQualifierAnalysis(Operation* op) {
  op->walk([&](Operation* op) {
    for (Value result : op->getResults()) {
      if (!isMutated(result)) {
        constValues.insert(result);
      }
    }
  });
}

}  // namespace heir
}  // namespace mlir
