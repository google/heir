#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"

#include <string>

#include "lib/Dialect/Debug/IR/DebugOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"       // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_ADDDEBUGPORT
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

FailureOr<Type> getPrivateKeyType(func::FuncOp op) {
  const auto* type = llvm::find_if(op.getArgumentTypes(), [](Type type) {
    return isa<LWECiphertextType>(getElementTypeOrSelf(type));
  });

  if (type == op.getArgumentTypes().end()) {
    return op.emitError(
        "Function does not have an argument of LWECiphertextType");
  }

  auto lweCiphertextType = cast<LWECiphertextType>(getElementTypeOrSelf(*type));

  auto lwePrivateKeyType =
      LWESecretKeyType::get(op.getContext(), lweCiphertextType.getKey(),
                            lweCiphertextType.getCiphertextSpace().getRing());
  return lwePrivateKeyType;
}

func::FuncOp getOrCreateExternalDebugFunc(
    ModuleOp module, Type lwePrivateKeyType, Type valueType,
    const DenseMap<Type, int>& typeToInt) {
  std::string funcName =
      "__heir_debug_" + std::to_string(typeToInt.at(valueType));

  auto* context = module.getContext();
  auto lookup = module.lookupSymbol<func::FuncOp>(funcName);
  if (lookup) return lookup;

  auto debugFuncType =
      FunctionType::get(context, {lwePrivateKeyType, valueType}, {});

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  auto funcOp = func::FuncOp::create(b, funcName, debugFuncType);
  // required for external func call
  funcOp.setPrivate();
  return funcOp;
}

void insertValidationOps(func::FuncOp op) {
  int count = 0;
  auto insertValidate = [&](Value value, OpBuilder& b) {
    Type valueType = value.getType();
    if (isa<LWECiphertextType>(getElementTypeOrSelf(valueType))) {
      debug::ValidateOp::create(b, value.getLoc(), value,
                                "heir_debug_" + std::to_string(count++),
                                nullptr);
    }
  };

  Block& entryBlock = op.getBody().getBlocks().front();
  OpBuilder argBuilder(&entryBlock, entryBlock.begin());
  for (auto arg : op.getArguments()) {
    insertValidate(arg, argBuilder);
  }

  op.walk([&](Operation* walkOp) {
    if (walkOp == op.getOperation() ||
        walkOp->hasTrait<OpTrait::IsTerminator>())
      return;
    OpBuilder opBuilder(walkOp->getBlock(), ++walkOp->getIterator());
    for (Value result : walkOp->getResults()) {
      insertValidate(result, opBuilder);
    }
  });
}

LogicalResult lowerValidationOps(func::FuncOp op, Value privateKey,
                                 int messageSize) {
  auto module = op->getParentOfType<ModuleOp>();
  DenseMap<Type, int> typeToInt;
  Type lwePrivateKeyType = privateKey.getType();

  auto walkResult = op.walk([&](debug::ValidateOp validateOp) {
    Value value = validateOp.getInput();
    Type valueType = value.getType();
    if (isa<LWECiphertextType>(getElementTypeOrSelf(valueType))) {
      if (!typeToInt.count(valueType)) {
        typeToInt[valueType] = typeToInt.size();
      }

      ImplicitLocOpBuilder b(validateOp.getLoc(), validateOp);
      SmallVector<NamedAttribute> attrs;
      // Transfer metadata from validateOp to CallOp
      attrs.push_back(b.getNamedAttr("debug.name", validateOp.getNameAttr()));
      if (validateOp.getMetadata()) {
        attrs.push_back(
            b.getNamedAttr("debug.metadata", validateOp.getMetadataAttr()));
      }
      attrs.push_back(b.getNamedAttr(
          "message.size", b.getStringAttr(std::to_string(messageSize))));

      auto debugFunc = getOrCreateExternalDebugFunc(module, lwePrivateKeyType,
                                                    valueType, typeToInt);
      Value privateKeyToPass = privateKey;
      if (privateKeyToPass.getType() !=
          debugFunc.getFunctionType().getInput(0)) {
        privateKeyToPass =
            UnrealizedConversionCastOp::create(
                b, validateOp.getLoc(), debugFunc.getFunctionType().getInput(0),
                privateKeyToPass)
                .getResult(0);
      }

      Value valueToPass = value;
      if (valueToPass.getType() != debugFunc.getFunctionType().getInput(1)) {
        valueToPass = b.create<UnrealizedConversionCastOp>(
                           validateOp.getLoc(),
                           debugFunc.getFunctionType().getInput(1), valueToPass)
                          .getResult(0);
      }

      auto callOp = b.create<func::CallOp>(
          debugFunc, ArrayRef<Value>{privateKeyToPass, valueToPass});
      callOp->setDialectAttrs(attrs);

      validateOp.erase();
    }
    return WalkResult::advance();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

LogicalResult convertFunc(func::FuncOp op, int messageSize,
                          bool insertDebugAfterEveryOp) {
  FailureOr<Type> type = failure();
  if (insertDebugAfterEveryOp) {
    type = getPrivateKeyType(op);
    if (succeeded(type)) {
      insertValidationOps(op);
    }
  }

  // Check if there are any validation ops to lower before adding private key
  bool hasValidationOps = false;
  op.walk([&](debug::ValidateOp) {
    hasValidationOps = true;
    return WalkResult::interrupt();
  });

  if (!hasValidationOps) return success();

  // If we haven't inferred the type yet, do it now because we need it.
  if (failed(type)) {
    type = getPrivateKeyType(op);
  }

  // Check if private key is already an argument
  Value privateKey;
  for (auto arg : op.getArguments()) {
    if (isa<LWESecretKeyType>(arg.getType())) {
      privateKey = arg;
      break;
    }
  }

  if (!privateKey) {
    if (failed(type)) {
      // This case should only happen if validations were already present in a
      // function without a private key and no way to infer it.
      return success();
    }
    auto lwePrivateKeyType = type.value();
    if (failed(op.insertArgument(0, lwePrivateKeyType, nullptr, op.getLoc()))) {
      return op.emitError("failed to insert private key argument");
    }
    privateKey = op.getArgument(0);
  }

  if (failed(lowerValidationOps(op, privateKey, messageSize))) {
    return op.emitError("failed to lower validation ops");
  }
  return success();
}

struct AddDebugPort : impl::AddDebugPortBase<AddDebugPort> {
  using AddDebugPortBase::AddDebugPortBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    func::FuncOp entryFunc;
    if (!entryFunction.empty()) {
      entryFunc = module.lookupSymbol<func::FuncOp>(entryFunction);
    }

    if (entryFunc) {
      if (failed(
              convertFunc(entryFunc, messageSize, insertDebugAfterEveryOp))) {
        entryFunc->emitError("Failed to add debug port for func");
        signalPassFailure();
      }
      return;
    }

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isExternal()) continue;
      if (failed(convertFunc(funcOp, messageSize, insertDebugAfterEveryOp))) {
        funcOp->emitError("Failed to add debug port for func");
        signalPassFailure();
        return;
      }
    }
  }
};
}  // namespace lwe
}  // namespace heir
}  // namespace mlir
