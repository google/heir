#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"

#include <string>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_ADDDEBUGPORT
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

FailureOr<Type> getPrivateKeyType(func::FuncOp op) {
  const auto *type = llvm::find_if(op.getArgumentTypes(), [](Type type) {
    return mlir::isa<NewLWECiphertextType>(type);
  });

  if (type == op.getArgumentTypes().end()) {
    return op.emitError(
        "Function does not have an argument of LWECiphertextType");
  }

  auto lweCiphertextType = cast<NewLWECiphertextType>(*type);

  auto lwePrivateKeyType = NewLWESecretKeyType::get(
      op.getContext(), lweCiphertextType.getKey(),
      lweCiphertextType.getCiphertextSpace().getRing());
  return lwePrivateKeyType;
}

func::FuncOp getOrCreateExternalDebugFunc(
    ModuleOp module, Type lwePrivateKeyType,
    NewLWECiphertextType lweCiphertextType,
    const DenseMap<Type, int> &typeToInt) {
  std::string funcName =
      "__heir_debug_" + std::to_string(typeToInt.at(lweCiphertextType));

  auto *context = module.getContext();
  auto lookup = module.lookupSymbol<func::FuncOp>(funcName);
  if (lookup) return lookup;

  auto debugFuncType =
      FunctionType::get(context, {lwePrivateKeyType, lweCiphertextType}, {});

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  auto funcOp = b.create<func::FuncOp>(funcName, debugFuncType);
  // required for external func call
  funcOp.setPrivate();
  return funcOp;
}

LogicalResult insertExternalCall(func::FuncOp op, Type lwePrivateKeyType) {
  auto module = op->getParentOfType<ModuleOp>();

  // map ciphertext type to unique int
  DenseMap<Type, int> typeToInt;

  // implicit assumption the first argument is private key
  auto privateKey = op.getArgument(0);

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  op.walk([&](Operation *op) {
    b.setInsertionPointAfter(op);
    for (Value result : op->getResults()) {
      Type resultType = result.getType();
      if (auto lweCiphertextType = dyn_cast<NewLWECiphertextType>(resultType)) {
        // update typeToInt
        if (!typeToInt.count(resultType)) {
          typeToInt[resultType] = typeToInt.size();
        }
        b.create<func::CallOp>(
            getOrCreateExternalDebugFunc(module, lwePrivateKeyType,
                                         lweCiphertextType, typeToInt),
            ArrayRef<Value>{privateKey, result});
      }
    }
    return WalkResult::advance();
  });
  return success();
}

LogicalResult convertFunc(func::FuncOp op) {
  auto type = getPrivateKeyType(op);
  if (failed(type)) return failure();
  auto lwePrivateKeyType = type.value();

  op.insertArgument(0, lwePrivateKeyType, nullptr, op.getLoc());
  if (failed(insertExternalCall(op, lwePrivateKeyType))) {
    return failure();
  }
  return success();
}

struct AddDebugPort : impl::AddDebugPortBase<AddDebugPort> {
  using AddDebugPortBase::AddDebugPortBase;

  void runOnOperation() override {
    auto result =
        getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
          if (op.getSymName() == entryFunction && failed(convertFunc(op))) {
            op->emitError("Failed to add client interface for func");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) signalPassFailure();
  }
};
}  // namespace lwe
}  // namespace heir
}  // namespace mlir
