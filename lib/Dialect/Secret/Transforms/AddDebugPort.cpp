#include "lib/Dialect/Secret/Transforms/AddDebugPort.h"

#include <string>

#include "lib/Dialect/FuncUtils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

func::FuncOp getOrCreateExternalDebugFunc(ModuleOp module, Type valueType) {
  std::string funcName = "__heir_debug_";

  SmallString<16> buffer;
  SmallString<16> buffer2;
  llvm::raw_svector_ostream os(buffer);
  valueType.print(os);
  funcName += sanitizeIdentifier(buffer, buffer2);

  auto *context = module.getContext();
  auto lookup = module.lookupSymbol<func::FuncOp>(funcName);
  if (lookup) return lookup;

  auto debugFuncType = FunctionType::get(context, {valueType}, {});

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  auto funcOp = b.create<func::FuncOp>(funcName, debugFuncType);
  // required for external func call
  funcOp.setPrivate();
  return funcOp;
}

LogicalResult insertExternalCall(secret::GenericOp op) {
  auto module = op->getParentOfType<ModuleOp>();

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), op.getBody());

  auto insertCall = [&](Value value) {
    Type valueType = value.getType();

    b.create<func::CallOp>(getOrCreateExternalDebugFunc(module, valueType),
                           ArrayRef<Value>{value});
  };

  // insert for each argument
  for (auto arg : op.getBody()->getArguments()) {
    insertCall(arg);
  }

  // insert after each op
  op.walk([&](Operation *op) {
    if (mlir::isa<secret::GenericOp>(op)) {
      return;
    }

    b.setInsertionPointAfter(op);
    for (Value result : op->getResults()) {
      insertCall(result);
    }
  });
  return success();
}

#define GEN_PASS_DEF_SECRETADDDEBUGPORT
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

struct AddDebugPort : impl::SecretAddDebugPortBase<AddDebugPort> {
  using SecretAddDebugPortBase::SecretAddDebugPortBase;

  void runOnOperation() override {
    getOperation()->walk([&](secret::GenericOp genericOp) {
      if (failed(insertExternalCall(genericOp))) {
        genericOp->emitError("Failed to add debug port for genericOp");
        signalPassFailure();
      }
    });
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
