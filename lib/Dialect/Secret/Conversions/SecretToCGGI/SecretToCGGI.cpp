#include "lib/Dialect/Secret/Conversions/SecretToCGGI/SecretToCGGI.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/Sequence.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "secret-to-cggi"

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOCGGI
#include "lib/Dialect/Secret/Conversions/SecretToCGGI/SecretToCGGI.h.inc"

namespace {

// buildSelectTruthTable recursively creates arithmetic operations to compute a
// lookup table on plaintext integers.
Value buildSelectTruthTable(Location loc, OpBuilder &b, Value t, Value f,
                            const APInt &lut, ValueRange lutInputs) {
  int tableSize = lut.getBitWidth();
  assert(tableSize == (int)(1ull << lutInputs.size()));
  if (tableSize == 1) {
    return lut.isZero() ? f : t;
  }

  int halfSize = tableSize / 2;
  auto firstHalf = lut.extractBits(halfSize, 0);
  auto lastHalf = lut.extractBits(halfSize, halfSize);

  Value selectTrue =
      buildSelectTruthTable(loc, b, t, f, firstHalf, lutInputs.drop_back());
  Value selectFalse =
      buildSelectTruthTable(loc, b, t, f, lastHalf, lutInputs.drop_back());
  return b.create<arith::SelectOp>(loc, lutInputs.back(), selectTrue,
                                   selectFalse);
}

Operation *convertWriteOpInterface(
    Operation *op, SmallVector<Value> indices, Value valueToStore,
    TypedValue<MemRefType> toMemRef,
    ContextAwareConversionPatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  MemRefType toMemRefTy = toMemRef.getType();
  Type valueToStoreType = valueToStore.getType();
  return llvm::TypeSwitch<Type, Operation *>(valueToStoreType)
      // Plaintext integer into memref
      .Case<IntegerType>([&](auto valType) {
        auto ctTy = toMemRefTy.getElementType();
        auto encoding = cast<lwe::LWECiphertextType>(ctTy).getEncoding();
        auto ptxtTy = lwe::LWEPlaintextType::get(b.getContext(), encoding);

        if (valType.getWidth() == 1) {
          auto ctValue = b.create<lwe::TrivialEncryptOp>(
              ctTy, b.create<lwe::EncodeOp>(ptxtTy, valueToStore, encoding),
              lwe::LWEParamsAttr());
          return b.create<memref::StoreOp>(ctValue, toMemRef, indices);
        }

        // Get i-th bit of input and insert the bit into the memref of
        // ciphertexts.
        auto loop = b.create<mlir::affine::AffineForOp>(0, valType.getWidth());
        b.setInsertionPointToStart(loop.getBody());
        auto idx = loop.getInductionVar();

        auto one = b.create<arith::ConstantOp>(
            valType, rewriter.getIntegerAttr(valType, 1));
        auto shiftAmount = b.create<arith::IndexCastOp>(valType, idx);
        auto bitMask = b.create<arith::ShLIOp>(valType, one, shiftAmount);
        auto andOp = b.create<arith::AndIOp>(valueToStore, bitMask);
        auto shifted = b.create<arith::ShRSIOp>(andOp, shiftAmount);
        auto bitValue = b.create<arith::TruncIOp>(b.getI1Type(), shifted);
        auto ctValue = b.create<lwe::TrivialEncryptOp>(
            ctTy, b.create<lwe::EncodeOp>(ptxtTy, bitValue, encoding),
            lwe::LWEParamsAttr());

        indices.push_back(idx);
        return b.create<memref::StoreOp>(ctValue, toMemRef, indices);
      })
      .Case<lwe::LWECiphertextType>([&](auto valType) {
        return b.create<memref::StoreOp>(valueToStore, toMemRef, indices);
      })
      .Case<MemRefType>([&](MemRefType valType) {
        int rank = toMemRefTy.getRank();

        // A store op with a memref value to store must have
        // originated from a secret encoding a multi-bit value. Under type
        // conversion, the op is storing a memref<BITSIZE!ct_ty> into a
        // memref of the form memref<?xBITSIZE!ct_ty>.

        // Subview the storage memref into a rank-reduced memref at the
        // storage indices.
        //   * Offset: offset at the storage index [indices, 0]
        //   * Strides: match original memref [1, 1, ..., 1]
        //   * Sizes: rank-reduce to the last dim [1, 1, ..., BITSIZE]
        SmallVector<OpFoldResult> offsets = getAsOpFoldResult(indices);
        offsets.push_back(OpFoldResult(b.getIndexAttr(0)));

        Attribute oneIdxAttr = rewriter.getIndexAttr(1);
        SmallVector<OpFoldResult> strides(rank, oneIdxAttr);
        SmallVector<OpFoldResult> sizes(rank - 1, oneIdxAttr);
        sizes.push_back(rewriter.getIndexAttr(toMemRefTy.getShape()[rank - 1]));

        mlir::Type memRefType =
            mlir::memref::SubViewOp::inferRankReducedResultType(
                valType.getShape(), toMemRefTy, offsets, sizes, strides);
        auto subview = b.create<memref::SubViewOp>(
            cast<MemRefType>(memRefType), toMemRef, offsets, sizes, strides);
        return b.create<memref::CopyOp>(valueToStore, subview);
      });
  llvm_unreachable("expected integer or memref to store in ciphertext memref");
}

Operation *convertReadOpInterface(
    Operation *op, SmallVector<Value> indices, Value fromMemRef,
    Type outputType, ContextAwareConversionPatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  MemRefType outputMemRefType = dyn_cast<MemRefType>(outputType);
  MemRefType fromMemRefType = cast<MemRefType>(fromMemRef.getType());
  int rank = fromMemRefType.getRank();

  // A load op with a memref value to store must have
  // originated from a secret encoding a multi-bit value. Under type
  // conversion, the op is loading a memref<BITSIZE!ct_ty> from a
  // memref<?xBITSIZE!ct_ty>.

  // Subview the storage memref into a rank-reduced memref at the
  // storage indices.
  //   * Offset: offset at the storage index [indices, 0]
  //   * Strides: match original memref [1, 1, ..., 1]
  //   * Sizes: rank-reduce to the last dim [1, 1, ..., BITSIZE]
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets = getAsOpFoldResult(indices);
  offsets.push_back(OpFoldResult(b.getIndexAttr(0)));
  SmallVector<OpFoldResult> strides(rank, oneIdxAttr);
  SmallVector<OpFoldResult> sizes(rank - 1, oneIdxAttr);
  sizes.push_back(rewriter.getIndexAttr(fromMemRefType.getShape()[rank - 1]));

  // We need to calculate the resulting subview shape which may have a dynamic
  // offset.
  mlir::Type memRefType = mlir::memref::SubViewOp::inferRankReducedResultType(
      outputMemRefType.getShape(), fromMemRefType, offsets, sizes, strides);
  // If the offsets are dynamic and the resulting type does not match the
  // converted output type, we must allocate and copy into one with a static 0
  // offset. Otherwise we can return the subview.
  auto subViewOp = b.create<memref::SubViewOp>(
      cast<MemRefType>(memRefType), fromMemRef, offsets, sizes, strides);
  if (memRefType != outputMemRefType) {
    auto allocOp = b.create<memref::AllocOp>(outputMemRefType);
    b.create<memref::CopyOp>(subViewOp, allocOp);
    return allocOp;
  }
  return subViewOp;
}

SmallVector<Value> encodeInputs(
    Operation *op, ValueRange inputs,
    ContextAwareConversionPatternRewriter &rewriter) {
  // Get the ciphertext type.
  lwe::LWECiphertextType ctxtTy;
  for (auto input : inputs) {
    if (isa<lwe::LWECiphertextType>(input.getType())) {
      ctxtTy = cast<lwe::LWECiphertextType>(input.getType());
      break;
    }
  }

  // Encode any plaintexts in the inputs.
  auto encoding = cast<lwe::LWECiphertextType>(ctxtTy).getEncoding();
  auto ptxtTy = lwe::LWEPlaintextType::get(rewriter.getContext(), encoding);
  return llvm::to_vector(llvm::map_range(inputs, [&](auto input) -> Value {
    if (!isa<lwe::LWECiphertextType>(input.getType())) {
      IntegerType integerTy = dyn_cast<IntegerType>(input.getType());
      assert(integerTy && integerTy.getWidth() == 1 &&
             "LUT inputs should be single-bit integers");
      return rewriter
          .create<lwe::TrivialEncryptOp>(
              op->getLoc(), ctxtTy,
              rewriter.create<lwe::EncodeOp>(op->getLoc(), ptxtTy, input,
                                             encoding),
              lwe::LWEParamsAttr())
          .getResult();
    }
    return input;
  }));
}

}  // namespace

class SecretTypeConverter : public ContextAwareTypeConverter {
 public:
  SecretTypeConverter(MLIRContext *ctx, int minBitWidth)
      : minBitWidth(minBitWidth) {
    // Convert secret types to LWE ciphertext types
    addConversion([](Type type, Attribute attr) { return type; });
    addConversion([ctx, this](secret::SecretType type, Attribute attr) -> Type {
      return getLWECiphertextForInt(ctx, type.getValueType());
    });
  }

  Type getLWECiphertextForInt(MLIRContext *ctx, Type type) const {
    if (IntegerType intType = dyn_cast<IntegerType>(type)) {
      if (intType.getWidth() == 1) {
        return lwe::LWECiphertextType::get(
            ctx, lwe::UnspecifiedBitFieldEncodingAttr::get(ctx, minBitWidth),
            lwe::LWEParamsAttr());
      }
      return MemRefType::get(
          {intType.getWidth()},
          getLWECiphertextForInt(ctx, IntegerType::get(ctx, 1)));
    }
    ShapedType shapedType = dyn_cast<ShapedType>(type);
    assert(shapedType &&
           "expected shaped secret type for a non-integer secret");
    assert(isa<IntegerType>(shapedType.getElementType()) &&
           "expected integer element types for shaped secret types");
    auto elementType = getLWECiphertextForInt(ctx, shapedType.getElementType());
    SmallVector<int64_t> newShape = {shapedType.getShape().begin(),
                                     shapedType.getShape().end()};
    if (auto elementShape = dyn_cast<ShapedType>(elementType)) {
      // Flatten the element shape with the original shape
      newShape.insert(newShape.end(), elementShape.getShape().begin(),
                      elementShape.getShape().end());
      return MemRefType::get(newShape, elementShape.getElementType());
    }
    return shapedType.cloneWith(newShape, elementType);
  }

  inline Attribute defaultContext(MLIRContext *ctx) const {
    return UnitAttr::get(ctx);
  }

  FailureOr<Attribute> getContextualAttr(Value value) const override {
    // No attribute is necessary yet, since we aren't attaching contextual
    // information to the types yet.
    return defaultContext(value.getContext());
  }

  LogicalResult convertFuncSignature(
      FunctionOpInterface funcOp, SmallVectorImpl<Type> &newArgTypes,
      SmallVectorImpl<Type> &newResultTypes) const override {
    for (auto argType : funcOp.getArgumentTypes()) {
      newArgTypes.push_back(
          convertType(argType, defaultContext(funcOp.getContext())));
    }

    for (auto resultType : funcOp.getResultTypes()) {
      newResultTypes.push_back(
          convertType(resultType, defaultContext(funcOp.getContext())));
    }

    return success();
  }

  int minBitWidth;
};

class SecretGenericOpLUTConversion
    : public SecretGenericOpConversion<comb::TruthTableOp, cggi::Lut3Op> {
  using SecretGenericOpConversion<comb::TruthTableOp,
                                  cggi::Lut3Op>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> encodedInputs =
        encodeInputs(op.getOperation(), inputs, rewriter);

    // Assemble the lookup table.
    comb::TruthTableOp truthOp =
        cast<comb::TruthTableOp>(op.getBody()->getOperations().front());
    return rewriter
        .replaceOpWithNewOp<cggi::Lut3Op>(op, encodedInputs[0],
                                          encodedInputs[1], encodedInputs[2],
                                          truthOp.getLookupTable())
        .getOperation();
  }
};

class SecretGenericOpMemRefLoadConversion
    : public SecretGenericOpConversion<memref::LoadOp> {
  using SecretGenericOpConversion<memref::LoadOp>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    memref::LoadOp loadOp =
        cast<memref::LoadOp>(op.getBody()->getOperations().front());
    if (auto lweType = dyn_cast<lwe::LWECiphertextType>(outputTypes[0])) {
      return rewriter
          .replaceOpWithNewOp<memref::LoadOp>(op, inputs[0],
                                              loadOp.getIndices())
          .getOperation();
    }
    auto *newOp = convertReadOpInterface(loadOp, loadOp.getIndices(), inputs[0],
                                         outputTypes[0], rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

class SecretGenericOpMemRefAllocConversion
    : public SecretGenericOpConversion<memref::AllocOp> {
  using SecretGenericOpConversion<memref::AllocOp>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    // Preserve the alignment attribute.
    auto innerOp = cast<memref::AllocOp>(op.getBody()->getOperations().front());
    SmallVector<NamedAttribute> newAttributes(attributes.begin(),
                                              attributes.end());
    if (innerOp.getAlignmentAttr()) {
      newAttributes.push_back(NamedAttribute(
          "alignment", cast<Attribute>(innerOp.getAlignmentAttr())));
    }
    return rewriter
        .replaceOpWithNewOp<memref::AllocOp>(op, outputTypes, inputs,
                                             newAttributes)
        .getOperation();
  }
};

class SecretGenericOpMemRefCollapseShapeConversion
    : public SecretGenericOpConversion<memref::CollapseShapeOp> {
  using SecretGenericOpConversion<
      memref::CollapseShapeOp>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    // Preserve the reassociation attribute.
    auto innerOp =
        cast<memref::CollapseShapeOp>(op.getBody()->getOperations().front());
    SmallVector<NamedAttribute> newAttributes(attributes.begin(),
                                              attributes.end());
    newAttributes.push_back(
        NamedAttribute("reassociation", innerOp.getReassociationAttr()));
    return rewriter
        .replaceOpWithNewOp<memref::CollapseShapeOp>(op, outputTypes, inputs,
                                                     newAttributes)
        .getOperation();
  }
};

template <typename GateOp, typename CGGIGateOp>
class SecretGenericOpGateConversion
    : public SecretGenericOpConversion<GateOp, CGGIGateOp> {
  using SecretGenericOpConversion<GateOp,
                                  CGGIGateOp>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    return rewriter
        .replaceOpWithNewOp<CGGIGateOp>(
            op, outputTypes, encodeInputs(op.getOperation(), inputs, rewriter),
            attributes)
        .getOperation();
  }
};

using SecretGenericOpInvConversion =
    SecretGenericOpGateConversion<comb::InvOp, cggi::NotOp>;
using SecretGenericOpAndConversion =
    SecretGenericOpGateConversion<comb::AndOp, cggi::AndOp>;
using SecretGenericOpOrConversion =
    SecretGenericOpGateConversion<comb::OrOp, cggi::OrOp>;
using SecretGenericOpNorConversion =
    SecretGenericOpGateConversion<comb::NorOp, cggi::NorOp>;
using SecretGenericOpXNorConversion =
    SecretGenericOpGateConversion<comb::XNorOp, cggi::XNorOp>;
using SecretGenericOpXorConversion =
    SecretGenericOpGateConversion<comb::XorOp, cggi::XorOp>;
using SecretGenericOpNandConversion =
    SecretGenericOpGateConversion<comb::NandOp, cggi::NandOp>;

class SecretGenericOpAffineLoadConversion
    : public SecretGenericOpConversion<affine::AffineLoadOp> {
  using SecretGenericOpConversion<
      affine::AffineLoadOp>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    affine::AffineLoadOp loadOp =
        cast<affine::AffineLoadOp>(op.getBody()->getOperations().front());
    if (auto lweType = dyn_cast<lwe::LWECiphertextType>(outputTypes[0])) {
      return rewriter
          .replaceOpWithNewOp<affine::AffineLoadOp>(
              op, inputs[0], loadOp.getAffineMap(), loadOp.getIndices())
          .getOperation();
    }
    auto indices = affine::expandAffineMap(
        rewriter, op.getLoc(), loadOp.getAffineMap(), loadOp.getIndices());
    if (!indices) {
      op.emitError() << "expected affine access indices";
    }
    auto *newOp = convertReadOpInterface(
        loadOp, {indices.value().begin(), indices.value().end()}, inputs[0],
        outputTypes[0], rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

class SecretGenericOpAffineStoreConversion
    : public SecretGenericOpConversion<affine::AffineStoreOp, memref::StoreOp> {
  using SecretGenericOpConversion<affine::AffineStoreOp,
                                  memref::StoreOp>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    affine::AffineStoreOp storeOp =
        cast<affine::AffineStoreOp>(op.getBody()->getOperations().front());
    auto toMemRef = cast<TypedValue<MemRefType>>(inputs[1]);
    auto indices = affine::expandAffineMap(
        rewriter, op.getLoc(), storeOp.getAffineMap(), storeOp.getIndices());
    if (!indices) {
      op.emitError() << "expected affine access indices";
    }
    auto *newOp = convertWriteOpInterface(
        storeOp, {indices.value().begin(), indices.value().end()}, inputs[0],
        toMemRef, rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

class SecretGenericOpMemRefStoreConversion
    : public SecretGenericOpConversion<memref::StoreOp> {
  using SecretGenericOpConversion<memref::StoreOp>::SecretGenericOpConversion;

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    memref::StoreOp storeOp =
        cast<memref::StoreOp>(op.getBody()->getOperations().front());
    auto toMemRef = cast<TypedValue<MemRefType>>(inputs[1]);
    auto *newOp = convertWriteOpInterface(
        storeOp, {storeOp.getIndices().begin(), storeOp.getIndices().end()},
        inputs[0], toMemRef, rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

// ConvertTruthTableOp converts truth table ops with fully plaintext values.
struct ConvertTruthTableOp
    : public ContextAwareOpConversionPattern<comb::TruthTableOp> {
  ConvertTruthTableOp(mlir::MLIRContext *context)
      : ContextAwareOpConversionPattern<comb::TruthTableOp>(context) {}

  using ContextAwareOpConversionPattern::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      comb::TruthTableOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3) {
      op->emitError() << "expected 3 truth table arguments to lower to CGGI";
    }
    // A plaintext truth table op should not be contained inside a generic.
    assert(op->getParentOfType<secret::GenericOp>() == nullptr);

    // Create a truth table op out of arithmetic statements.
    Value t = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    Value f = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
    rewriter.replaceOp(op, buildSelectTruthTable(op.getLoc(), rewriter, t, f,
                                                 op.getLookupTable().getValue(),
                                                 adaptor.getInputs()));

    return success();
  }
};

// ConvertSecretCastOp removes secret.cast operations between multi-bit secret
// integers and tensors of single-bit secrets.
struct ConvertSecretCastOp
    : public ContextAwareOpConversionPattern<secret::CastOp> {
  ConvertSecretCastOp(mlir::MLIRContext *context)
      : ContextAwareOpConversionPattern<secret::CastOp>(context) {}

  using ContextAwareOpConversionPattern::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::CastOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    // The original secret cast reconciles multi-bit or memrefs of values like
    // secret<i8> to secret<memref<8xi1>> or secret<memref<2x4xi2>> and
    // secret<memref<16xi1>> and vice versa. One of these will always be a
    // flattened memref<Nxi1> type that Yosys uses as input/output types. After
    // secret-to-cggi type conversion, both will be memref types.
    auto lhsMemRefTy = dyn_cast<MemRefType>(adaptor.getInput().getType());
    auto rhsMemRefTy = dyn_cast<MemRefType>(typeConverter->convertType(
        op.getOutput().getType(),
        typeConverter->getContextualAttr(op.getOutput()).value_or(nullptr)));

    if (!lhsMemRefTy || !rhsMemRefTy) {
      return op->emitOpError()
             << "expected cast between multi-bit secret types, got "
             << op.getInput().getType() << " and " << op.getOutput().getType();
    }

    int lhsBits = lhsMemRefTy.getNumElements();
    int rhsBits = rhsMemRefTy.getNumElements();
    if (lhsBits != rhsBits) {
      return op->emitOpError()
             << "expected cast between secrets holding the "
                "same number of total bits, got "
             << op.getInput().getType() << " and " << op.getOutput().getType();
    }

    // Fold the cast if the resulting type converted shapes are the same.
    if (lhsMemRefTy.getShape() == rhsMemRefTy.getShape()) {
      // This happens when one of the original shapes was a memref and the other
      // was an integer, for e.g. a secret.cast from i8 to memref<8xlwe_ct>.
      assert(isa<ShapedType>(op.getInput().getType().getValueType()) !=
             isa<ShapedType>(op.getOutput().getType().getValueType()));
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Otherwise, resolve the cast by reshaping the input. This happens when the
    // original operand or result was a memref which Yosys optimizer flattened,
    // for e.g. secret<memref<2xi4>> (memref<8xlwe_ct>).
    if (lhsMemRefTy && rhsMemRefTy) {
      if (lhsMemRefTy.getRank() > rhsMemRefTy.getRank() &&
          rhsMemRefTy.getRank() == 1) {
        // This case happens when converting a high dimension memref into a flat
        // memref of single bits as an operand to a Yosys optimized body, e.g.
        // secret<memref<1x1xi8>> (memref<1x1x8xlwe_ciphertext>) to
        // secret<memref<8xi1>> (memref<8xlwe_ciphertext>). In this case,
        // collapse the memref shape.
        SmallVector<mlir::ReassociationIndices> reassociation;
        auto range = llvm::seq<unsigned>(0, lhsMemRefTy.getRank());
        reassociation.emplace_back(range.begin(), range.end());
        rewriter.replaceOpWithNewOp<memref::CollapseShapeOp>(
            op, adaptor.getInput(), reassociation);
        return success();
      } else if (lhsMemRefTy.getRank() < rhsMemRefTy.getRank() &&
                 lhsMemRefTy.getRank() == 1) {
        // This is the case of converting results of Yosys optimized bodies,
        // flat memrefs of single bits, into a high dimensional memref, e.g.
        // secret<memref<8xi1>> (memref<8xlwe_ciphertext>) to
        // secret<memref<1x2xi4>> (memref<1x2x4xlwe_ciphertext>).
        SmallVector<mlir::ReassociationIndices> reassociation;
        auto range = llvm::seq<unsigned>(0, rhsMemRefTy.getRank());
        reassociation.emplace_back(range.begin(), range.end());
        rewriter.replaceOpWithNewOp<memref::ExpandShapeOp>(
            op, rhsMemRefTy, adaptor.getInput(), reassociation);
        return success();
      } else if (lhsMemRefTy.getShape() != rhsMemRefTy.getShape()) {
        // Fallback to a reinterpret cast to resolve the memref shapes.
        int64_t offset;
        SmallVector<int64_t> strides;
        if (failed(rhsMemRefTy.getStridesAndOffset(strides, offset)))
          return rewriter.notifyMatchFailure(
              op, "failed to get stride and offset exprs");
        auto castOp = rewriter.create<memref::ReinterpretCastOp>(
            op.getLoc(), rhsMemRefTy, adaptor.getInput(), offset,
            rhsMemRefTy.getShape(), strides);
        rewriter.replaceOp(op, castOp);
        return success();
      }
    }

    return failure();
  }
};

// ConvertSecretConcealOp lowers secret.conceal to a series of trivial_encrypt
// ops stored into a memref.
struct ConvertSecretConcealOp
    : public ContextAwareOpConversionPattern<secret::ConcealOp> {
  ConvertSecretConcealOp(mlir::MLIRContext *context)
      : ContextAwareOpConversionPattern<secret::ConcealOp>(context) {}

  using ContextAwareOpConversionPattern::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type convertedTy = getTypeConverter()->convertType(
        op.getResult().getType(), getTypeConverter()
                                      ->getContextualAttr(op.getResult())
                                      .value_or(nullptr));
    auto memrefTy = dyn_cast<MemRefType>(convertedTy);
    auto ctTy = cast<lwe::LWECiphertextType>(memrefTy.getElementType());
    auto ptxtTy =
        lwe::LWEPlaintextType::get(rewriter.getContext(), ctTy.getEncoding());

    Value valueToStore = adaptor.getCleartext();
    IntegerType valueType = dyn_cast<IntegerType>(valueToStore.getType());

    auto allocOp = b.create<memref::AllocOp>(memrefTy);
    Value valueMemref = allocOp.getResult();

    if (valueType.getWidth() == 1) {
      auto ctValue = b.create<lwe::TrivialEncryptOp>(
          ctTy,
          b.create<lwe::EncodeOp>(ptxtTy, valueToStore, ctTy.getEncoding()),
          lwe::LWEParamsAttr());
      SmallVector<Value> indices = {b.create<arith::ConstantIndexOp>(0)};
      b.create<memref::StoreOp>(ctValue, valueMemref, indices);
    } else {
      auto loop = b.create<mlir::affine::AffineForOp>(0, valueType.getWidth());
      b.setInsertionPointToStart(loop.getBody());
      auto idx = loop.getInductionVar();

      auto one = b.create<arith::ConstantOp>(
          valueType, rewriter.getIntegerAttr(valueType, 1));
      auto shiftAmount = b.create<arith::IndexCastOp>(valueType, idx);
      auto bitMask = b.create<arith::ShLIOp>(valueType, one, shiftAmount);
      auto andOp = b.create<arith::AndIOp>(valueToStore, bitMask);
      auto shifted = b.create<arith::ShRSIOp>(andOp, shiftAmount);
      auto bitValue = b.create<arith::TruncIOp>(b.getI1Type(), shifted);
      auto ctValue = b.create<lwe::TrivialEncryptOp>(
          ctTy, b.create<lwe::EncodeOp>(ptxtTy, bitValue, ctTy.getEncoding()),
          lwe::LWEParamsAttr());

      SmallVector<Value> indices = {idx};
      b.create<memref::StoreOp>(ctValue, valueMemref, indices);
    }

    rewriter.replaceAllOpUsesWith(op, valueMemref);
    rewriter.eraseOp(op);

    return success();
  }
};

struct ResolveUnrealizedConversionCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  ResolveUnrealizedConversionCast(mlir::MLIRContext *context)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

int findLUTSize(MLIRContext *context, Operation *module) {
  int maxIntSize = 1;
  auto processOperation = [&](Operation *op) {
    if (isa<comb::CombDialect>(op->getDialect())) {
      int currentSize = 0;
      if (dyn_cast<comb::TruthTableOp>(op))
        currentSize = 3;
      else
        currentSize = op->getResults().getTypes()[0].getIntOrFloatBitWidth();

      maxIntSize = std::max(maxIntSize, currentSize);
    }
  };

  // Walk all operations within the module in post-order (default)
  module->walk(processOperation);

  return maxIntSize;
}

struct SecretToCGGI : public impl::SecretToCGGIBase<SecretToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    // Helper for future lowerings that want to know what scheme was used
    module->setAttr(kCGGISchemeAttrName, UnitAttr::get(context));

    int lutSize = findLUTSize(context, module);

    SecretTypeConverter typeConverter(context, lutSize);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<cggi::CGGIDialect, arith::ArithDialect,
                           lwe::LWEDialect, memref::MemRefDialect>();
    target.addIllegalOp<comb::TruthTableOp, secret::CastOp, secret::GenericOp,
                        secret::ConcealOp>();

    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      // Legal only when the memref element type matches the stored
      // type.
      return op.getMemRefType().getElementType() ==
             op.getValueToStore().getType();
    });
    target.addDynamicallyLegalOp<affine::AffineStoreOp>(
        [&](affine::AffineStoreOp op) {
          // Legal only when the memref element type matches the stored
          // type.
          return op.getMemRefType().getElementType() ==
                 op.getValueToStore().getType();
        });
    target.addDynamicallyLegalOp<memref::LoadOp>([&](memref::LoadOp op) {
      // Legal only when the memref element type matches the loaded type.
      return op.getMemRefType().getElementType() == op.getResult().getType();
    });
    target.addDynamicallyLegalOp<affine::AffineLoadOp>(
        [&](affine::AffineLoadOp op) {
          // Legal only when the memref element type matches the loaded
          // type.
          return op.getMemRefType().getElementType() ==
                 op.getResult().getType();
        });

    // The conversion of an op whose result is an iter_arg of affine.for can
    // result in a type mismatch between the affine.for op and its own internal
    // block arguments. So we need to add a pattern that at least converts the
    // block signature of a for op's body.
    target.addDynamicallyLegalOp<affine::AffineForOp>(
        [&](affine::AffineForOp op) {
          return typeConverter.isLegal(op) &&
                 typeConverter.isLegal(&op.getBodyRegion());
        });
    target.addDynamicallyLegalOp<affine::AffineYieldOp>(
        [&](auto op) { return typeConverter.isLegal(op); });

    patterns.add<
        SecretGenericOpLUTConversion, SecretGenericOpMemRefAllocConversion,
        SecretGenericOpConversion<memref::DeallocOp, memref::DeallocOp>,
        SecretGenericOpMemRefCollapseShapeConversion,
        SecretGenericOpMemRefLoadConversion,
        SecretGenericOpAffineStoreConversion,
        SecretGenericOpAffineLoadConversion,
        SecretGenericOpMemRefStoreConversion, ConvertTruthTableOp,
        SecretGenericOpInvConversion, SecretGenericOpAndConversion,
        SecretGenericOpNorConversion, SecretGenericOpNandConversion,
        SecretGenericOpOrConversion, SecretGenericOpXNorConversion,
        SecretGenericOpXorConversion, ConvertSecretCastOp,
        ConvertSecretConcealOp, ConvertAnyContextAware<affine::AffineForOp>,
        ConvertAnyContextAware<affine::AffineYieldOp>>(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyContextAwarePartialConversion(module, target,
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet cleanupPatterns(context);
    patterns.add<ResolveUnrealizedConversionCast>(context);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
