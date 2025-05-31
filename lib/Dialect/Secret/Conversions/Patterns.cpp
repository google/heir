#include "lib/Dialect/Secret/Conversions/Patterns.h"

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

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
    return parentFunc->emitError()
           << "expected secret.conceal op to be inside a function with a "
              "single LWE ciphertext return type; it may be that "
              "the type converter failed to run on this func "
              "because the mgmt attribute is missing.";
  }

  auto *ctx = op->getContext();
  Type encryptionKeyType = usePublicKey
                               ? (Type)lwe::NewLWEPublicKeyType::get(
                                     ctx, lwe::KeyAttr::get(ctx, 0),
                                     resultCtTy.getCiphertextSpace().getRing())
                               : (Type)lwe::NewLWESecretKeyType::get(
                                     ctx, lwe::KeyAttr::get(ctx, 0),
                                     resultCtTy.getCiphertextSpace().getRing());
  Value keyBlockArg =
      insertKeyArgument(parentFunc, encryptionKeyType, rewriter);

  auto plaintextTy = lwe::NewLWEPlaintextType::get(
      op.getContext(), resultCtTy.getApplicationData(),
      resultCtTy.getPlaintextSpace());
  auto encoded = rewriter.create<lwe::RLWEEncodeOp>(
      op.getLoc(), plaintextTy, adaptor.getCleartext(),
      resultCtTy.getPlaintextSpace().getEncoding(),
      resultCtTy.getPlaintextSpace().getRing());
  auto encryptOp = rewriter.create<lwe::RLWEEncryptOp>(
      op.getLoc(), resultCtTy, encoded.getResult(), keyBlockArg);

  // Copy attributes from the original op to preserve any mgmt attrs needed by
  // dialect conversion from secret to scheme.
  encryptOp->setAttrs(op->getAttrs());

  rewriter.replaceOp(op, encryptOp);
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
  auto encryptionKeyType = lwe::NewLWESecretKeyType::get(
      ctx, lwe::KeyAttr::get(ctx, 0), argCtTy.getCiphertextSpace().getRing());
  Value keyBlockArg =
      insertKeyArgument(parentFunc, encryptionKeyType, rewriter);

  auto plaintextTy = lwe::NewLWEPlaintextType::get(op.getContext(),
                                                   argCtTy.getApplicationData(),
                                                   argCtTy.getPlaintextSpace());
  auto decrypted = rewriter.create<lwe::RLWEDecryptOp>(
      op.getLoc(), plaintextTy, adaptor.getInput(), keyBlockArg);

  // Note: we use the secret.reveal op's original result type as the result
  // type for the new rlwe_decode op, rather than the type from the parent
  // func, because the client helper includes extra ops that convert from a
  // plaintext type to a cleartext type. This may ultimately raise questions
  // about the purpose of the rlwe_encode/rlwe_decode ops, but the remaining
  // part of the RLWE encoding that does not have any associated op is the
  // NTT/iNTT (for evaluation/slot encoding/decoding) and in the case that
  // remains, the output of the encoding step is a tensor of a specific size,
  // even when the original cleartext data might be a scalar or a tensor of a
  // smaller size.
  //
  // For example, for this input IR:
  //
  //  func.func @dot_product__decrypt__result0(
  //      %arg0: !secret.secret<tensor<8xi16>>) -> i16 {
  //    %c0 = arith.constant 0 : index
  //    %0 = secret.reveal %arg0 : !secret.secret<tensor<8xi16>> ->
  //    tensor<8xi16> %extracted = tensor.extract %0[%c0] : tensor<8xi16> return
  //    %extracted : i16
  //  }
  //
  // this pattern lowers the reveal op to have an output tensor type (rather
  // than i16). The rest of the IR manages unpacking, and the rlwe_decode op
  // will only manage the cryptosystem-relevant decoding step (such as iNTT).
  auto decoded = rewriter.create<lwe::RLWEDecodeOp>(
      op.getLoc(), op.getResult().getType(), decrypted.getResult(),
      argCtTy.getPlaintextSpace().getEncoding(),
      argCtTy.getPlaintextSpace().getRing());

  rewriter.replaceOp(op, decoded);
  return success();
}

}  // namespace heir
}  // namespace mlir
