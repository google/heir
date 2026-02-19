#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"

#include <string>

#include "lib/Dialect/Debug/IR/DebugOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/TransformUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_ADDDEBUGPORT
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

FailureOr<Type> getPrivateKeyType(func::FuncOp op) {
  const auto* type = llvm::find_if(op.getArgumentTypes(), [](Type type) {
    return mlir::isa<LWECiphertextType>(getElementTypeOrSelf(type));
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
    ModuleOp module, Type lwePrivateKeyType,
    LWECiphertextType lweCiphertextType, const DenseMap<Type, int>& typeToInt) {
  std::string funcName =
      "__heir_debug_" + std::to_string(typeToInt.at(lweCiphertextType));

  auto* context = module.getContext();
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

void insertValidationOps(func::FuncOp op) {
  int count = 0;
  auto insertValidate = [&](Value value, OpBuilder& b) {
    Type valueType = value.getType();
    if (mlir::isa<LWECiphertextType>(valueType)) {
      b.create<debug::ValidateOp>(value.getLoc(), value,
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

LogicalResult lowerValidationOps(func::FuncOp op, Type lwePrivateKeyType,
                                 int messageSize) {
  auto module = op->getParentOfType<ModuleOp>();
  DenseMap<Type, int> typeToInt;
  Value privateKey = op.getArgument(0);

  auto walkResult = op.walk([&](debug::ValidateOp validateOp) {
    Value value = validateOp.getInput();
    Type valueType = value.getType();
    if (auto lweCiphertextType = dyn_cast<LWECiphertextType>(valueType)) {
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

      auto callOp = b.create<func::CallOp>(
          getOrCreateExternalDebugFunc(module, lwePrivateKeyType,
                                       lweCiphertextType, typeToInt),
          ArrayRef<Value>{privateKey, value});
      callOp->setDialectAttrs(attrs);

      validateOp.erase();
    }
    return WalkResult::advance();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

LogicalResult convertFunc(func::FuncOp op, int messageSize,
                          bool insertDebugAfterEveryOp) {
  if (insertDebugAfterEveryOp) {
    insertValidationOps(op);
  }

  // Check if there are any validation ops to lower before adding private key
  bool hasValidationOps = false;
  op.walk([&](debug::ValidateOp) {
    hasValidationOps = true;
    return WalkResult::interrupt();
  });

  if (!hasValidationOps) return success();

  auto type = getPrivateKeyType(op);
  if (failed(type)) return op.emitError("failed to get private key type");
  auto lwePrivateKeyType = type.value();

  if (failed(op.insertArgument(0, lwePrivateKeyType, nullptr, op.getLoc()))) {
    return op.emitError("failed to insert private key argument");
  }

  if (failed(lowerValidationOps(op, lwePrivateKeyType, messageSize))) {
    return op.emitError("failed to lower validation ops");
  }
  return success();
}

struct AddDebugPort : impl::AddDebugPortBase<AddDebugPort> {
  using AddDebugPortBase::AddDebugPortBase;

  void runOnOperation() override {
    auto funcOp =
        detectEntryFunction(cast<ModuleOp>(getOperation()), entryFunction);
    if (funcOp &&
        failed(convertFunc(funcOp, messageSize, insertDebugAfterEveryOp))) {
      funcOp->emitError("Failed to add debug port for func");
      signalPassFailure();
    }
  }
};
}  // namespace lwe
}  // namespace heir
}  // namespace mlir
