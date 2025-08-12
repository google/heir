#include "lib/Dialect/Mgmt/Transforms/Utils.h"

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"
#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult copyMgmtAttrToClientHelpers(Operation* op) {
  auto& kArgMgmtAttrName = mgmt::MgmtDialect::kArgMgmtAttrName;

  ModuleOp moduleOp = cast<ModuleOp>(op);
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

    Operation* maybeFunc = moduleOp.lookupSymbol(originalFuncName);
    if (!maybeFunc) {
      op->emitError() << "module missing func with name " << originalFuncName;
      return WalkResult::interrupt();
    }

    func::FuncOp originalFunc = cast<func::FuncOp>(maybeFunc);
    int index = cast<IntegerAttr>(attr.get(kClientHelperIndex)).getInt();

    auto shouldPropagate = [&](Type type) {
      return isa<secret::SecretType>(type);
    };

    if (clientEncAttr) {
      auto mgmtAttr = originalFunc.getArgAttr(index, kArgMgmtAttrName);
      if (!mgmtAttr) {
        originalFunc.emitError()
            << "expected mgmt attribute on original function argument";
        return WalkResult::interrupt();
      }
      funcOp.setResultAttr(0, kArgMgmtAttrName, mgmtAttr);
      backwardPropagateAnnotation(funcOp, kArgMgmtAttrName, shouldPropagate);
    } else {
      auto mgmtAttr = originalFunc.getResultAttr(index, kArgMgmtAttrName);
      if (!mgmtAttr) {
        originalFunc.emitError()
            << "expected mgmt attribute on original function argument";
        return WalkResult::interrupt();
      }
      funcOp.setArgAttr(0, kArgMgmtAttrName, mgmtAttr);
      forwardPropagateAnnotation(funcOp, kArgMgmtAttrName, shouldPropagate);
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

}  // namespace heir
}  // namespace mlir
