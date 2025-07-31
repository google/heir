#include "lib/Transforms/AddClientInterface/AddClientInterface.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Support/WalkResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"   // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "add-client-interface"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ADDCLIENTINTERFACE
#include "lib/Transforms/AddClientInterface/AddClientInterface.h.inc"

using secret::ConcealOp;
using secret::RevealOp;
using secret::SecretType;
using tensor_ext::AssignLayoutOp;
using tensor_ext::OriginalTypeAttr;
using tensor_ext::TensorExtDialect;
using tensor_ext::UnpackOp;

auto &kOriginalTypeAttrName = TensorExtDialect::kOriginalTypeAttrName;

namespace {
Type stripSecretType(Type type) {
  if (auto secretType = dyn_cast<SecretType>(type))
    return secretType.getValueType();
  return type;
}
}  // namespace

/// Generates an encryption func for one types.
LogicalResult generateEncryptionFunc(func::FuncOp op,
                                     BlockArgument funcArgument,
                                     ImplicitLocOpBuilder &builder,
                                     int64_t ciphertextSize,
                                     bool enableLayoutAssignment) {
  auto insertionBlock = builder.getInsertionBlock();
  auto insertionPoint = builder.getInsertionPoint();
  std::string encFuncName("");
  llvm::raw_string_ostream encNameOs(encFuncName);
  encNameOs << op.getSymName() << "__encrypt__arg"
            << funcArgument.getArgNumber();
  auto originalTypeAttr = op.getArgAttrOfType<OriginalTypeAttr>(
      funcArgument.getArgNumber(), kOriginalTypeAttrName);
  if (enableLayoutAssignment && !originalTypeAttr) {
    return op.emitError() << "function argument at index "
                          << funcArgument.getArgNumber()
                          << " missing original type attribute";
  }
  Type encArgType = originalTypeAttr ? originalTypeAttr.getOriginalType()
                                     : stripSecretType(funcArgument.getType());
  Type encReturnType = funcArgument.getType();

  FunctionType encFuncType =
      FunctionType::get(builder.getContext(), {encArgType}, {encReturnType});
  auto encFuncOp = func::FuncOp::create(builder, encFuncName, encFuncType);

  encFuncOp->setAttr(
      kClientEncFuncAttrName,
      builder.getDictionaryAttr({
          builder.getNamedAttr(kClientHelperFuncName,
                               builder.getStringAttr(op.getSymName())),
          builder.getNamedAttr(
              kClientHelperIndex,
              builder.getI64IntegerAttr(funcArgument.getArgNumber())),
      }));

  Block *entryBlock = encFuncOp.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);
  IRRewriter b(builder);

  SmallVector<Value> encValuesToReturn;
  auto operand = encFuncOp.getArgument(0);

  // If the output is encrypted, we need to encode and encrypt
  if (auto resultCtTy = dyn_cast<SecretType>(encReturnType)) {
    Value valueToEncrypt = operand;
    if (enableLayoutAssignment) {
      // Apply the layout from the original type, which handles the job of
      // converting a scalar to a tensor, and out a tensor in a
      // ciphertext-semantic tensor so that it can then be encoded/encrypted.
      // Note that this op still needs to be lowered, which implies the
      // relevant pattern from convert-to-ciphertext-semantics must be run
      // after this pass.
      auto assignLayoutOp = AssignLayoutOp::create(
          builder, operand, originalTypeAttr.getLayout());
      auto res = implementAssignLayout(assignLayoutOp, ciphertextSize, builder,
                                       [&](Operation *createdOp) {});
      if (failed(res)) return failure();
      b.replaceOp(assignLayoutOp, res.value());
      valueToEncrypt = res.value();
    }
    auto encrypted = ConcealOp::create(builder, valueToEncrypt);
    encValuesToReturn.push_back(encrypted.getResult());
  } else {
    // Otherwise, return the input unchanged.
    encValuesToReturn.push_back(operand);
  }

  func::ReturnOp::create(builder, encValuesToReturn);
  builder.setInsertionPoint(insertionBlock, insertionPoint);
  return success();
}

/// Generates a decryption func for one type.
LogicalResult generateDecryptionFunc(func::FuncOp op, Type decFuncArgType,
                                     int originalFuncReturnIndex,
                                     ImplicitLocOpBuilder &builder,
                                     bool enableLayoutAssignment) {
  auto insertionBlock = builder.getInsertionBlock();
  auto insertionPoint = builder.getInsertionPoint();
  std::string decFuncName("");
  llvm::raw_string_ostream encNameOs(decFuncName);
  encNameOs << op.getSymName() << "__decrypt__result"
            << originalFuncReturnIndex;
  auto originalTypeAttr = op.getResultAttrOfType<OriginalTypeAttr>(
      originalFuncReturnIndex, kOriginalTypeAttrName);
  Type originalType = originalTypeAttr ? originalTypeAttr.getOriginalType()
                                       : stripSecretType(decFuncArgType);
  assert(op.getFunctionType().getResult(originalFuncReturnIndex) ==
             decFuncArgType &&
         "Decryption func arg type must match original func return type");
  if (enableLayoutAssignment && !originalTypeAttr) {
    return op.emitError() << "function return value at index "
                          << originalFuncReturnIndex
                          << " missing original type attribute";
  }
  SmallVector<Type> funcArgTypes = {decFuncArgType};
  FunctionType decFuncType =
      FunctionType::get(builder.getContext(), {decFuncArgType}, {originalType});
  auto decFuncOp = func::FuncOp::create(builder, decFuncName, decFuncType);

  decFuncOp->setAttr(
      kClientDecFuncAttrName,
      builder.getDictionaryAttr({
          builder.getNamedAttr(kClientHelperFuncName,
                               builder.getStringAttr(op.getSymName())),
          builder.getNamedAttr(
              kClientHelperIndex,
              builder.getI64IntegerAttr(originalFuncReturnIndex)),
      }));

  builder.setInsertionPointToEnd(decFuncOp.addEntryBlock());
  IRRewriter b(builder);
  SmallVector<Value> decValuesToReturn;

  // If the input is ciphertext, we need to decrypt and unpack
  if (auto argCtTy = dyn_cast<SecretType>(decFuncArgType)) {
    auto decrypted = RevealOp::create(builder, decFuncOp.getArgument(0));
    if (!enableLayoutAssignment) {
      decValuesToReturn.push_back(decrypted.getResult());
    } else {
      Type dataSemanticType = originalTypeAttr.getOriginalType();
      auto unpackOp = tensor_ext::UnpackOp::create(
          builder, dataSemanticType, decrypted.getResult(),
          cast<tensor_ext::LayoutAttr>(originalTypeAttr.getLayout()));

      Value res =
          implementUnpackOp(unpackOp, builder, [&](Operation *createdOp) {});
      b.replaceOp(unpackOp, res);
      decValuesToReturn.push_back(res);
    }
  } else {
    // Otherwise, return the input unchanged.
    decValuesToReturn.push_back(decFuncOp.getArgument(0));
  }

  func::ReturnOp::create(builder, decValuesToReturn);
  builder.setInsertionPoint(insertionBlock, insertionPoint);
  return success();
}

/// Adds the client interface for a single func. This should only be used on the
/// "entry" func for the IR being compiled, but there may be multiple.
LogicalResult convertFunc(func::FuncOp op, int64_t ciphertextSize,
                          bool enableLayoutAssignment) {
  if (op.isDeclaration()) {
    LLVM_DEBUG(op->emitWarning("Skipping client interface for external func"));
    return success();
  }

  auto module = op->getParentOfType<ModuleOp>();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());
  builder.setInsertionPointAfter(op);

  // We need one encryption function per argument and one decryption
  // function per return value. This is mainly to avoid complicated C++ codegen
  // when encrypting multiple inputs which requires out-params.
  for (BlockArgument val : op.getArguments()) {
    auto argTy = val.getType();
    if (auto argCtTy = dyn_cast<SecretType>(argTy)) {
      if (failed(generateEncryptionFunc(op, val, builder, ciphertextSize,
                                        enableLayoutAssignment))) {
        return failure();
      }
    }
  }

  ArrayRef<Type> returnTypes = op.getFunctionType().getResults();
  for (size_t i = 0; i < returnTypes.size(); ++i) {
    auto returnTy = returnTypes[i];
    if (auto returnCtTy = dyn_cast<SecretType>(returnTy)) {
      if (failed(generateDecryptionFunc(op, returnCtTy, i, builder,
                                        enableLayoutAssignment))) {
        return failure();
      }
    }
  }
  LLVM_DEBUG(module.dump());

  return success();
}

struct AddClientInterface : impl::AddClientInterfaceBase<AddClientInterface> {
  using AddClientInterfaceBase::AddClientInterfaceBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto result = root->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
      if (failed(convertFunc(op, ciphertextSize, enableLayoutAssignment))) {
        op->emitError("Failed to add client interface for func");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
