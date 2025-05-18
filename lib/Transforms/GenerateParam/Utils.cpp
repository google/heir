#include "lib/Transforms/GenerateParam/Utils.h"

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project

namespace mlir {
namespace heir {

// Client helper functions require mgmt attributes in order for the type
// converter to convert them. The correct mgmt attribute to use is
// determined by the function the helpers are made for, and so
// this method copies the relevant mgmt attributes from
// the original function to the client helper function arg/result attrs
// so they can be propagated through those IRs. Any existing mgmt attributes
// on the client helpers are ignored.
//
// E.g., given the following functions
//
//   func.func @foo(
//      %arg0: !secret.secret<tensor<8xi16>> {mgmt.mgmt = ...})
//         -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = ...}) {
//     ...
//   }
//   func.func @foo__encrypt__arg0(
//      %arg0: tensor<8xi16>) -> !secret.secret<tensor<8xi16>>
//      attributes {client_enc_func = {func_name = "foo", index = 0 : i64}} {
//     ...
//   }
//   func.func @foo__decrypt__result0(
//      %arg0: !secret.secret<tensor<8xi16>>) -> i16
//       attributes {client_dec_func = {func_name = "foo", index = 0 : i64}} {
//     ...
//   }
//
// The encrypt function needs its result annotated with a mgmt attr that
// matches the mgmt attr of @foo's 0th argument, and the decrypt function's
// argument needs a mgmt attr that matches the mgmt attr of @foo's 0th result.
// Then the mgmt attr needs to be backward propagated in the encryption
// function and forward propagated in the decryption function.
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
