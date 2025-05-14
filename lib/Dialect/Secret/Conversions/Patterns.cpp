#include "lib/Dialect/Secret/Conversions/Patterns.h"

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

Value insertKeyArgument(func::FuncOp parentFunc, Type encryptionKeyType,
                        ContextAwareConversionPatternRewriter &rewriter) {
  // The new key type is inserted as the last argument of the parent function.
  auto oldFunctionType = parentFunc.getFunctionType();
  SmallVector<Type, 4> newInputTypes;
  newInputTypes.append(oldFunctionType.getInputs().begin(),
                       oldFunctionType.getInputs().end());
  newInputTypes.push_back(encryptionKeyType);
  auto newFunctionType =
      rewriter.getFunctionType(newInputTypes, oldFunctionType.getResults());

  Value keyBlockArg;
  rewriter.modifyOpInPlace(parentFunc, [&]() {
    parentFunc.setType(newFunctionType);
    keyBlockArg = parentFunc.getBody().addArgument(encryptionKeyType,
                                                   parentFunc.getLoc());
  });

  return keyBlockArg;
}

LogicalResult ConvertClientConceal::matchAndRewrite(
    secret::ConcealOp op, OpAdaptor adaptor,
    ContextAwareConversionPatternRewriter &rewriter) const {
  func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc || !parentFunc->hasAttr(kClientEncFuncAttrName)) {
    return op->emitError() << "expected to be inside a function with attribute "
                           << kClientEncFuncAttrName;
  }

  // The encryption func encrypts a single value, so it must have a single
  // return type. This relies on the ContextAwareFuncConversion to have already
  // run, so that the result type is type converted in-place.
  auto resultCtTy =
      dyn_cast<lwe::NewLWECiphertextType>(parentFunc.getResultTypes()[0]);
  if (!resultCtTy) {
    return op->emitError() << "expected to be inside a function with a single "
                           << "LWE ciphertext return type";
  }

  auto *ctx = op->getContext();
  Type encryptionKeyType = usePublicKey
                               ? (Type)lwe::NewLWEPublicKeyType::get(
                                     ctx, lwe::KeyAttr::get(ctx, 0), ring)
                               : (Type)lwe::NewLWESecretKeyType::get(
                                     ctx, lwe::KeyAttr::get(ctx, 0), ring);
  Value keyBlockArg =
      insertKeyArgument(parentFunc, encryptionKeyType, rewriter);

  auto plaintextTy = lwe::NewLWEPlaintextType::get(
      op.getContext(), resultCtTy.getApplicationData(),
      resultCtTy.getPlaintextSpace());
  auto encoded = rewriter.create<lwe::RLWEEncodeOp>(
      op.getLoc(), plaintextTy, adaptor.getCleartext(),
      resultCtTy.getPlaintextSpace().getEncoding(),
      resultCtTy.getPlaintextSpace().getRing());
  auto encrypted = rewriter.create<lwe::RLWEEncryptOp>(
      op.getLoc(), resultCtTy, encoded.getResult(), keyBlockArg);

  rewriter.replaceOp(op, encrypted);
  return success();
}

LogicalResult ConvertClientReveal::matchAndRewrite(
    secret::RevealOp op, OpAdaptor adaptor,
    ContextAwareConversionPatternRewriter &rewriter) const {
  func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc || !parentFunc->hasAttr(kClientDecFuncAttrName)) {
    return op->emitError() << "expected to be inside a function with attribute "
                           << kClientDecFuncAttrName;
  }

  // The decryption func decrypts a single value, so it must have a single
  // argument. This relies on the ContextAwareFuncConversion to have already
  // run, so that the argument type is type converted in-place.
  auto argCtTy =
      dyn_cast<lwe::NewLWECiphertextType>(parentFunc.getArgumentTypes()[0]);
  if (!argCtTy) {
    return op->emitError() << "expected to be inside a function with a single "
                           << "LWE ciphertext argument type";
  }

  auto *ctx = op->getContext();
  auto encryptionKeyType =
      lwe::NewLWESecretKeyType::get(ctx, lwe::KeyAttr::get(ctx, 0), ring);
  Value keyBlockArg =
      insertKeyArgument(parentFunc, encryptionKeyType, rewriter);
  Type resultTy = parentFunc.getResultTypes()[0];

  auto plaintextTy = lwe::NewLWEPlaintextType::get(op.getContext(),
                                                   argCtTy.getApplicationData(),
                                                   argCtTy.getPlaintextSpace());
  auto decrypted = rewriter.create<lwe::RLWEDecryptOp>(
      op.getLoc(), plaintextTy, adaptor.getInput(), keyBlockArg);
  auto decoded = rewriter.create<lwe::RLWEDecodeOp>(
      op.getLoc(), resultTy, decrypted.getResult(),
      argCtTy.getPlaintextSpace().getEncoding(),
      argCtTy.getPlaintextSpace().getRing());

  rewriter.replaceOp(op, decoded);
  return success();
}

}  // namespace heir
}  // namespace mlir
