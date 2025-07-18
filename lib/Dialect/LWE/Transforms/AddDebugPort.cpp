#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"

#include <string>

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/TransformUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
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
  auto funcOp = func::FuncOp::create(b, funcName, debugFuncType);
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

  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
      op.getLoc(), &op.getBody().getBlocks().front());

  auto insertCall = [&](Value value) {
    Type valueType = value.getType();
    // NOTE: this won't work for shaped input like tensor<2x!lwe.ciphertext>
    if (auto lweCiphertextType = dyn_cast<NewLWECiphertextType>(valueType)) {
      // update typeToInt
      if (!typeToInt.count(valueType)) {
        typeToInt[valueType] = typeToInt.size();
      }

      // get attribute associated with value
      SmallVector<NamedAttribute> attrs;
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        auto *parentOp = blockArg.getOwner()->getParentOp();
        auto funcOp = dyn_cast<FunctionOpInterface>(parentOp);
        if (funcOp) {
          // always dialect attr
          for (auto namedAttr : funcOp.getArgAttrs(blockArg.getArgNumber())) {
            attrs.push_back(namedAttr);
          }
        }
      } else {
        auto *parentOp = value.getDefiningOp();
        for (auto namedAttr : parentOp->getDialectAttrs()) {
          attrs.push_back(namedAttr);
        }
      }

      auto messageType =
          lweCiphertextType.getApplicationData().getMessageType();
      auto messageSize = 1;
      if (auto tensorMessageType = dyn_cast<TensorType>(messageType)) {
        auto shape = tensorMessageType.getShape();
        if (shape.size() != 1) {
          op->emitWarning("Only support 1D tensor for message type");
        }
        messageSize = shape[0];
      }
      attrs.push_back(b.getNamedAttr(
          "message.size", b.getStringAttr(std::to_string(messageSize))));

      func::CallOp::create(
          b,
          getOrCreateExternalDebugFunc(module, lwePrivateKeyType,
                                       lweCiphertextType, typeToInt),
          ArrayRef<Value>{privateKey, value})
          ->setDialectAttrs(attrs);
    }
  };

  // insert for each argument
  for (auto arg : op.getArguments()) {
    insertCall(arg);
  }

  // insert after each HE op
  op.walk([&](Operation *op) {
    b.setInsertionPointAfter(op);
    for (Value result : op->getResults()) {
      insertCall(result);
    }
    return WalkResult::advance();
  });
  return success();
}

LogicalResult convertFunc(func::FuncOp op) {
  auto type = getPrivateKeyType(op);
  if (failed(type)) return failure();
  auto lwePrivateKeyType = type.value();

  if (failed(op.insertArgument(0, lwePrivateKeyType, nullptr, op.getLoc()))) {
    return failure();
  }
  if (failed(insertExternalCall(op, lwePrivateKeyType))) {
    return failure();
  }
  return success();
}

struct AddDebugPort : impl::AddDebugPortBase<AddDebugPort> {
  using AddDebugPortBase::AddDebugPortBase;

  void runOnOperation() override {
    auto funcOp =
        detectEntryFunction(cast<ModuleOp>(getOperation()), entryFunction);
    if (funcOp && failed(convertFunc(funcOp))) {
      funcOp->emitError("Failed to configure the crypto context for func");
      signalPassFailure();
    }
  }
};
}  // namespace lwe
}  // namespace heir
}  // namespace mlir
