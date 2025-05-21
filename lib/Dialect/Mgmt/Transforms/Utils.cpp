#include "lib/Dialect/Mgmt/Transforms/Utils.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult copyAttrToClientHelpers(Operation *op, StringRef attrName) {
  ModuleOp moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    return op->emitError() << "copyAttrToClientHelpers expected module op";
  }
  WalkResult result = op->walk([&](func::FuncOp funcOp) {
    // Check for the helper attributes
    auto clientEncAttr =
        funcOp->getAttrOfType<mlir::DictionaryAttr>(kClientEncFuncAttrName);
    auto clientDecAttr =
        funcOp->getAttrOfType<mlir::DictionaryAttr>(kClientDecFuncAttrName);

    if (!clientEncAttr && !clientDecAttr) return WalkResult::advance();

    DictionaryAttr attr = clientEncAttr ? clientEncAttr : clientDecAttr;
    llvm::StringRef originalFuncName =
        cast<StringAttr>(attr.get(kClientHelperFuncName));
    func::FuncOp originalFunc =
        cast<func::FuncOp>(moduleOp.lookupSymbol(originalFuncName));
    int index = cast<IntegerAttr>(attr.get(kClientHelperIndex)).getInt();

    auto shouldPropagate = [&](Type type) {
      return isa<secret::SecretType>(type);
    };

    if (clientEncAttr) {
      auto mgmtAttr = originalFunc.getArgAttr(index, attrName);
      if (!mgmtAttr) {
        originalFunc.emitError()
            << "expected mgmt attribute on original function argument";
        return WalkResult::interrupt();
      }
      funcOp.setResultAttr(0, attrName, mgmtAttr);
      backwardPropagateAnnotation(funcOp, attrName, shouldPropagate);
    } else {
      auto mgmtAttr = originalFunc.getResultAttr(index, attrName);
      if (!mgmtAttr) {
        originalFunc.emitError()
            << "expected mgmt attribute on original function argument";
        return WalkResult::interrupt();
      }
      funcOp.setArgAttr(0, attrName, mgmtAttr);
      forwardPropagateAnnotation(funcOp, attrName, shouldPropagate);
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

}  // namespace heir
}  // namespace mlir
