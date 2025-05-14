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

void copyMgmtAttrToFunc(Operation *top) {
  top->walk([&](secret::GenericOp genericOp) {
    for (auto i = 0; i != genericOp->getNumOperands(); ++i) {
      auto operand = genericOp.getOperand(i);
      auto funcBlockArg = dyn_cast<BlockArgument>(operand);
      if (isa<secret::SecretType>(operand.getType()) && funcBlockArg) {
        auto funcOp =
            dyn_cast<func::FuncOp>(funcBlockArg.getOwner()->getParentOp());
        auto mgmtAttr =
            genericOp.getOperandAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName);
        if (mgmtAttr) {
          funcOp.setArgAttr(funcBlockArg.getArgNumber(),
                            mgmt::MgmtDialect::kArgMgmtAttrName, mgmtAttr);
        }
      }
    }
  });
  // some unused func secret type arg should also be annotated with mgmt attr,
  // inferred from other used arg
  top->walk([&](func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }
    Attribute firstMgmtAttr;
    for (auto i = 0; i != funcOp.getNumArguments(); ++i) {
      firstMgmtAttr = funcOp.getArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName);
      if (firstMgmtAttr) {
        break;
      }
    }
    for (auto i = 0; i != funcOp.getNumArguments(); ++i) {
      auto arg = funcOp.getArgument(i);
      if (firstMgmtAttr && mlir::isa<secret::SecretType>(arg.getType()) &&
          !funcOp.getArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName)) {
        funcOp.setArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName,
                          firstMgmtAttr);
      }
    }
  });

  // Now handle returned values -> func op result attrs
  copyReturnOperandAttrsToFuncResultAttrs(top,
                                          mgmt::MgmtDialect::kArgMgmtAttrName);
}

LogicalResult copyMgmtAttrToClientHelpers(Operation *op) {
  auto &kArgMgmtAttrName = mgmt::MgmtDialect::kArgMgmtAttrName;

  ModuleOp moduleOp = dyn_cast<ModuleOp>(op);
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
