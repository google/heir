#include "lib/Dialect/Secret/Conversions/Patterns.h"

#include <cstdint>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#define DEBUG_TYPE "secret-conversion-patterns"

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

// Assert the slice parameters are aligned with the ciphertext axis.
static LogicalResult validateSliceAlignment(
    Value tensorVal, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, Operation* op,
    ContextAwareConversionPatternRewriter& rewriter) {
  auto inputTy = dyn_cast<RankedTensorType>(tensorVal.getType());
  if (!inputTy) {
    return rewriter.notifyMatchFailure(op, "expected ranked tensor input");
  }

  int64_t lastDim = inputTy.getRank() - 1;

  if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2) {
    return rewriter.notifyMatchFailure(
        op, "expected slice to index a 2D tensor (ct x ciphertext)");
  }

  // The slice must start at the beginning of a ciphertext row.
  // E.g., offsets[-1] == 0.
  if (getConstantIntValue(offsets[lastDim]) != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "expected slice to start at the beginning of a ciphertext "
        "row");
  }

  // The slice must not skip any elements in the ciphertext row.
  // E.g., all strides = 1
  if (llvm::any_of(strides, [&](OpFoldResult ofr) {
        return getConstantIntValue(ofr) != 1;
      })) {
    return rewriter.notifyMatchFailure(op,
                                       "expected slice to have unit stride");
  }

  if (getConstantIntValue(sizes[0]) != 1) {
    return rewriter.notifyMatchFailure(
        op,
        "expected slice to include a full ciphertext row, but first sizes "
        "entry was not 1");
  }

  // The slice's size must match the ciphertext size.
  // E.g., sizes[-1] == 4096.
  if (inputTy.getDimSize(lastDim) != getConstantIntValue(sizes[lastDim])) {
    return rewriter.notifyMatchFailure(
        op, "expected slice to include a full ciphertext row");
  }

  return success();
}

// An extract_slice op corresponds to extracting one ciphertext from a tensor
// of ciphertexts. E.g., the input ciphertext-semantic tensor might be
// tensor<4x4096>, and the supported extract_slice op extracts one 4096-sized
// slice aligned with one row of the tensor.
FailureOr<Operation*> ConvertExtractSlice::matchAndRewriteInner(
    secret::GenericOp genericOp, TypeRange outputTypes, ValueRange inputs,
    ArrayRef<NamedAttribute> attributes,
    ContextAwareConversionPatternRewriter& rewriter) const {
  tensor::ExtractSliceOp op = cast<tensor::ExtractSliceOp>(
      genericOp.getBody()->getOperations().front());

  auto offsets = op.getMixedOffsets();
  auto sizes = op.getMixedSizes();
  auto strides = op.getMixedStrides();

  auto validationRes =
      validateSliceAlignment(inputs[0], offsets, sizes, strides, op, rewriter);
  if (failed(validationRes)) {
    return validationRes;
  }

  // Now convert the op to a tensor.extract op that extracts one ciphertext.
  OpFoldResult offset = offsets[0];
  Value offsetVal;
  if (auto idx = getConstantIntValue(offset); idx.has_value()) {
    offsetVal =
        arith::ConstantIndexOp::create(rewriter, op.getLoc(), idx.value());
  } else {
    offsetVal = cast<Value>(offset);
  }

  // Exactly one ciphertext output is guaranteed since validateSliceAlignment
  // ensures that the result size is exactly a ciphertext size
  auto resultCtTy = cast<lwe::LWECiphertextType>(outputTypes[0]);
  auto extractOp = tensor::ExtractOp::create(rewriter, op.getLoc(), resultCtTy,
                                             inputs[0], {offsetVal});
  rewriter.replaceOp(genericOp, extractOp);
  return extractOp.getOperation();
}

// An insert_slice op corresponds to inserting one ciphertext into a tensor
// of ciphertexts. E.g., the destination ciphertext-semantic tensor might be
// tensor<4x4096>, and the supported insert_slice op inserts one 4096-sized
// slice aligned with one row of the tensor.
FailureOr<Operation*> ConvertInsertSlice::matchAndRewriteInner(
    secret::GenericOp genericOp, TypeRange outputTypes, ValueRange inputs,
    ArrayRef<NamedAttribute> attributes,
    ContextAwareConversionPatternRewriter& rewriter) const {
  tensor::InsertSliceOp op =
      cast<tensor::InsertSliceOp>(genericOp.getBody()->getOperations().front());
  Value scalar = inputs[0];
  Value dest = inputs[1];
  auto offsets = op.getMixedOffsets();
  auto sizes = op.getMixedSizes();
  auto strides = op.getMixedStrides();

  auto validationRes = validateSliceAlignment(op.getDest(), offsets, sizes,
                                              strides, op, rewriter);
  if (failed(validationRes)) {
    return validationRes;
  }

  // Now convert the op to a tensor.insert op that inserts one ciphertext.
  SmallVector<Value> indices;
  for (size_t i = 0; i < offsets.size() - 1; ++i) {
    if (auto idx = getConstantIntValue(offsets[i]); idx.has_value()) {
      indices.push_back(
          arith::ConstantIndexOp::create(rewriter, op.getLoc(), idx.value()));
    } else {
      return rewriter.notifyMatchFailure(
          op, "expected insert_slice to have constant offsets");
    }
  }

  auto resultTensorOfCtsTy = cast<RankedTensorType>(outputTypes[0]);
  // This is a bit of a hack: if the dest tensor is (a mgmt.init of) a
  // tensor.empty, then it won't be type converted in the adaptor because it's
  // not an input to the secret.generic (even if another pattern handles the
  // tensor.empty properly). However, the tensor.empty needs to be converted to
  // a tensor.empty of ciphertext types. So just do the conversion here and make
  // a new tenosr.empty op.
  if (auto initOp = dyn_cast_or_null<mgmt::InitOp>(dest.getDefiningOp())) {
    if (auto emptyOp = dyn_cast_or_null<tensor::EmptyOp>(
            initOp.getOperand().getDefiningOp())) {
      dest = tensor::EmptyOp::create(rewriter, op.getLoc(),
                                     resultTensorOfCtsTy.getShape(),
                                     resultTensorOfCtsTy.getElementType());
      if (initOp.use_empty()) rewriter.eraseOp(initOp);
      if (emptyOp.use_empty()) rewriter.eraseOp(emptyOp);
    }
  }
  auto insertOp = tensor::InsertOp::create(
      rewriter, op.getLoc(), resultTensorOfCtsTy, scalar, dest, indices);
  rewriter.replaceOp(genericOp, insertOp);
  return insertOp.getOperation();
}

LogicalResult ConvertEmpty::matchAndRewrite(
    mgmt::InitOp op, OpAdaptor adaptor,
    ContextAwareConversionPatternRewriter& rewriter) const {
  if (!op.getOperand().getDefiningOp<tensor::EmptyOp>()) return failure();

  auto mgmtAttrResult = getTypeConverter()->getContextualAttr(op.getResult());
  if (failed(mgmtAttrResult)) return failure();

  RankedTensorType ciphertextType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(
          op.getResult().getType(), mgmtAttrResult.value()));
  LLVM_DEBUG(llvm::dbgs() << "cipherext type: " << ciphertextType << "\n");
  if (ciphertextType == nullptr)
    return rewriter.notifyMatchFailure(
        op, "failed to convert empty tensor type to tensor of ciphertext");

  rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, ciphertextType.getShape(),
                                               ciphertextType.getElementType());
  return success();
}

}  // namespace heir
}  // namespace mlir
