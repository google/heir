#include "lib/Dialect/Secret/Conversions/Patterns.h"

#include <cstdint>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

Value insertKeyArgument(func::FuncOp parentFunc, Type encryptionKeyType,
                        ContextAwareConversionPatternRewriter& rewriter) {
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
    ContextAwareConversionPatternRewriter& rewriter) const {
  func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc || !parentFunc->hasAttr(kClientEncFuncAttrName)) {
    return op->emitError() << "expected to be inside a function with attribute "
                           << kClientEncFuncAttrName;
  }

  // The encryption func encrypts a single value, so it must have a single
  // return type. This return type may be split over multiple ciphertexts. This
  // relies on the ContextAwareFuncConversion to have already run, so that the
  // result type is type converted in-place.
  auto resultCtTy = dyn_cast<lwe::LWECiphertextType>(
      getElementTypeOrSelf(parentFunc.getResultTypes()[0]));
  if (!resultCtTy) {
    return rewriter.notifyMatchFailure(
        op,
        "expected secret.conceal op to be inside a function with "
        " LWE ciphertexts return type; it may be that "
        "the type converter failed to run on this func "
        "because the mgmt attribute is missing.");
  }

  if (resultCtTy.getCiphertextSpace()
          .getRing()
          .getPolynomialModulus()
          .getPolynomial()
          .getDegree() == 1) {
    return op->emitError() << "Only RLWE ciphertexts are supported, "
                              "but detected a (scalar) LWE ciphertext";
  }

  auto* ctx = op->getContext();
  Type encryptionKeyType = usePublicKey
                               ? (Type)lwe::LWEPublicKeyType::get(
                                     ctx, lwe::KeyAttr::get(ctx, 0),
                                     resultCtTy.getCiphertextSpace().getRing())
                               : (Type)lwe::LWESecretKeyType::get(
                                     ctx, lwe::KeyAttr::get(ctx, 0),
                                     resultCtTy.getCiphertextSpace().getRing());
  Value keyBlockArg =
      insertKeyArgument(parentFunc, encryptionKeyType, rewriter);

  auto plaintextTy = lwe::LWEPlaintextType::get(op.getContext(),
                                                resultCtTy.getApplicationData(),
                                                resultCtTy.getPlaintextSpace());

  auto encryptFn = [&](Value cleartext) -> lwe::RLWEEncryptOp {
    auto encoded =
        lwe::RLWEEncodeOp::create(rewriter, op.getLoc(), plaintextTy, cleartext,
                                  resultCtTy.getPlaintextSpace().getEncoding(),
                                  resultCtTy.getPlaintextSpace().getRing());
    auto encryptOp = lwe::RLWEEncryptOp::create(
        rewriter, op.getLoc(), resultCtTy, encoded.getResult(), keyBlockArg);
    // Copy attributes from the original op to preserve any mgmt attrs needed by
    // dialect conversion from secret to scheme.
    encryptOp->setAttrs(op->getAttrs());
    return encryptOp;
  };

  // If this is a tensor, then build a loop nest to handle each element.
  // tensor<Nx1024xf32> -> tensor<Nxlwe>
  auto cleartextTensorTy =
      dyn_cast<RankedTensorType>(adaptor.getCleartext().getType());
  if (cleartextTensorTy && cleartextTensorTy.getRank() == 2) {
    SmallVector<Value> ciphertexts;
    auto extractedTy = RankedTensorType::get(
        {cleartextTensorTy.getDimSize(1)}, cleartextTensorTy.getElementType());
    SmallVector<OpFoldResult> sizes = {
        rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(cleartextTensorTy.getDimSize(1))};
    SmallVector<OpFoldResult> strides(2, rewriter.getIndexAttr(1));
    for (int64_t i = 0; i < cleartextTensorTy.getDimSize(0); ++i) {
      SmallVector<OpFoldResult> offsets = {rewriter.getIndexAttr(i),
                                           rewriter.getIndexAttr(0)};
      Value extracted = tensor::ExtractSliceOp::create(
          rewriter, op.getLoc(), extractedTy, adaptor.getCleartext(), offsets,
          sizes, strides);
      ciphertexts.push_back(encryptFn(extracted).getResult());
    }
    // Create tensor.from_elements op.
    auto fromElementsOp =
        tensor::FromElementsOp::create(rewriter, op.getLoc(), ciphertexts);
    fromElementsOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, fromElementsOp);
    return success();
  }

  // Otherwise, just encrypt the single value.
  auto encryptOp = encryptFn(adaptor.getCleartext());
  rewriter.replaceOp(op, encryptOp);
  return success();
}

LogicalResult ConvertClientReveal::matchAndRewrite(
    secret::RevealOp op, OpAdaptor adaptor,
    ContextAwareConversionPatternRewriter& rewriter) const {
  func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc || !parentFunc->hasAttr(kClientDecFuncAttrName)) {
    return op->emitError() << "expected to be inside a function with attribute "
                           << kClientDecFuncAttrName;
  }

  // The decryption func decrypts a single value, so it must have a single
  // argument that may be split over multiple ciphertexts. This relies on the
  // ContextAwareFuncConversion to have already run, so that the argument type
  // is type converted in-place.
  auto argCtTy = dyn_cast<lwe::LWECiphertextType>(
      getElementTypeOrSelf(parentFunc.getArgumentTypes()[0]));
  if (!argCtTy) {
    return rewriter.notifyMatchFailure(op,
                                       "expected to be inside a function with "
                                       "LWE ciphertexts");
  }

  if (argCtTy.getCiphertextSpace()
          .getRing()
          .getPolynomialModulus()
          .getPolynomial()
          .getDegree() == 1) {
    return op->emitError() << "Only RLWE ciphertexts are supported, "
                              "but detected a (scalar) LWE ciphertext";
  }

  auto* ctx = op->getContext();
  auto encryptionKeyType = lwe::LWESecretKeyType::get(
      ctx, lwe::KeyAttr::get(ctx, 0), argCtTy.getCiphertextSpace().getRing());
  Value keyBlockArg =
      insertKeyArgument(parentFunc, encryptionKeyType, rewriter);

  auto decryptFn = [&](Value ciphertext, Type resultTy) -> lwe::RLWEDecodeOp {
    auto plaintextTy = lwe::LWEPlaintextType::get(op.getContext(),
                                                  argCtTy.getApplicationData(),
                                                  argCtTy.getPlaintextSpace());
    auto decrypted = lwe::RLWEDecryptOp::create(
        rewriter, op.getLoc(), plaintextTy, ciphertext, keyBlockArg);
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
    //    tensor<8xi16> %extracted = tensor.extract %0[%c0] : tensor<8xi16>
    //    return %extracted : i16
    //  }
    //
    // this pattern lowers the reveal op to have an output tensor type (rather
    // than i16). The rest of the IR manages unpacking, and the rlwe_decode op
    // will only manage the cryptosystem-relevant decoding step (such as iNTT).
    auto decoded = lwe::RLWEDecodeOp::create(
        rewriter, op.getLoc(), resultTy, decrypted.getResult(),
        argCtTy.getPlaintextSpace().getEncoding(),
        argCtTy.getPlaintextSpace().getRing());
    return decoded;
  };

  // If this is a tensor, then build a loop nest to handle each element.
  // tensor<Nxlwe> -> tensor<Nx1024xf32>
  auto ciphertextTensorTy =
      dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (ciphertextTensorTy) {
    if (ciphertextTensorTy.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected 1-D tensors (ct x ciphertext) for ciphertexts");
    }
    SmallVector<Value> cleartexts;
    for (int64_t i = 0; i < ciphertextTensorTy.getDimSize(0); ++i) {
      Value extracted = tensor::ExtractOp::create(
          rewriter, op.getLoc(), adaptor.getInput(),
          {arith::ConstantIndexOp::create(rewriter, op.getLoc(), i)});
      // The result of the decryption of one element should be tensor<1024xf32>
      auto resultTensorTy = cast<RankedTensorType>(op.getResult().getType());
      auto resultTy = RankedTensorType::get({1, resultTensorTy.getDimSize(1)},
                                            resultTensorTy.getElementType());
      cleartexts.push_back(decryptFn(extracted, resultTy).getResult());
    }
    // Create tensor.concat op.
    auto concatop = tensor::ConcatOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), /*dim=*/0, cleartexts);
    rewriter.replaceOp(op, concatop);
    return success();
  }

  rewriter.replaceOp(op,
                     decryptFn(adaptor.getInput(), op.getResult().getType()));
  return success();
}

}  // namespace heir
}  // namespace mlir
