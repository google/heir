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
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/MemrefToArith/Utils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/Sequence.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
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
Value buildSelectTruthTable(Location loc, OpBuilder& b, Value t, Value f,
                            const APInt& lut, ValueRange lutInputs) {
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
  return arith::SelectOp::create(b, loc, lutInputs.back(), selectTrue,
                                 selectFalse);
}

Operation* convertWriteOpInterface(Operation* op, SmallVector<Value> indices,
                                   Value valueToStore,
                                   TypedValue<MemRefType> toMemRef,
                                   ConversionPatternRewriter& rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  MemRefType toMemRefTy = toMemRef.getType();
  Type valueToStoreType = valueToStore.getType();
  return llvm::TypeSwitch<Type, Operation*>(valueToStoreType)
      // Plaintext integer into memref
      .Case<IntegerType>([&](auto valType) {
        auto ctTy = cast<lwe::LWECiphertextType>(toMemRefTy.getElementType());
        auto ptxtTy = lwe::LWEPlaintextType::get(b.getContext(),
                                                 ctTy.getApplicationData(),
                                                 ctTy.getPlaintextSpace());
        auto plaintextBits =
            rewriter.getIndexAttr(ctTy.getPlaintextSpace()
                                      .getRing()
                                      .getCoefficientType()
                                      .getIntOrFloatBitWidth());
        auto ciphertextBits =
            rewriter.getIndexAttr(ctTy.getCiphertextSpace()
                                      .getRing()
                                      .getCoefficientType()
                                      .getIntOrFloatBitWidth());

        if (valType.getWidth() == 1) {
          auto ctValue = lwe::TrivialEncryptOp::create(
              b, ctTy,
              lwe::EncodeOp::create(b, ptxtTy, valueToStore, plaintextBits,
                                    ctTy.getApplicationData().getOverflow()),
              ciphertextBits);
          return memref::StoreOp::create(b, ctValue, toMemRef, indices);
        }

        // Get i-th bit of input and insert the bit into the memref of
        // ciphertexts.
        auto loop = mlir::affine::AffineForOp::create(b, 0, valType.getWidth());
        b.setInsertionPointToStart(loop.getBody());
        auto idx = loop.getInductionVar();

        auto one = arith::ConstantOp::create(
            b, valType, rewriter.getIntegerAttr(valType, 1));
        auto shiftAmount = arith::IndexCastOp::create(b, valType, idx);
        auto bitMask = arith::ShLIOp::create(b, valType, one, shiftAmount);
        auto andOp = arith::AndIOp::create(b, valueToStore, bitMask);
        auto shifted = arith::ShRSIOp::create(b, andOp, shiftAmount);
        auto bitValue = arith::TruncIOp::create(b, b.getI1Type(), shifted);
        auto ctValue = lwe::TrivialEncryptOp::create(
            b, ctTy,
            lwe::EncodeOp::create(b, ptxtTy, bitValue, plaintextBits,
                                  ctTy.getApplicationData().getOverflow()),
            ciphertextBits);

        indices.push_back(idx);
        return memref::StoreOp::create(b, ctValue, toMemRef, indices);
      })
      .Case<lwe::LWECiphertextType>([&](auto valType) {
        return memref::StoreOp::create(b, valueToStore, toMemRef, indices);
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
        auto subview = memref::SubViewOp::create(
            b, cast<MemRefType>(memRefType), toMemRef, offsets, sizes, strides);
        return memref::CopyOp::create(b, valueToStore, subview);
      });
  llvm_unreachable("expected integer or memref to store in ciphertext memref");
}

Operation* convertReadOpInterface(Operation* op, SmallVector<Value> indices,
                                  Value fromMemRef, Type outputType,
                                  ConversionPatternRewriter& rewriter) {
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
  auto subViewOp = memref::SubViewOp::create(
      b, cast<MemRefType>(memRefType), fromMemRef, offsets, sizes, strides);
  if (memRefType != outputMemRefType) {
    auto allocOp = memref::AllocOp::create(b, outputMemRefType);
    memref::CopyOp::create(b, subViewOp, allocOp);
    return allocOp;
  }
  return subViewOp;
}

SmallVector<Value> encodeInputs(Operation* op, ValueRange inputs,
                                ConversionPatternRewriter& rewriter) {
  // Get the ciphertext type.
  lwe::LWECiphertextType ctxtTy;
  for (auto input : inputs) {
    if (isa<lwe::LWECiphertextType>(input.getType())) {
      ctxtTy = cast<lwe::LWECiphertextType>(input.getType());
      break;
    }
  }

  // Encode any plaintexts in the inputs.
  auto overflow = ctxtTy.getApplicationData().getOverflow();
  auto ptxtSpace = ctxtTy.getPlaintextSpace();
  auto plaintextBits = rewriter.getIndexAttr(
      ptxtSpace.getRing().getCoefficientType().getIntOrFloatBitWidth());
  auto ciphertextBits = rewriter.getIndexAttr(ctxtTy.getCiphertextSpace()
                                                  .getRing()
                                                  .getCoefficientType()
                                                  .getIntOrFloatBitWidth());
  auto ptxtTy = lwe::LWEPlaintextType::get(
      rewriter.getContext(), ctxtTy.getApplicationData(), ptxtSpace);
  return llvm::to_vector(llvm::map_range(inputs, [&](auto input) -> Value {
    if (!isa<lwe::LWECiphertextType>(input.getType())) {
      IntegerType integerTy = dyn_cast<IntegerType>(input.getType());
      assert(integerTy && integerTy.getWidth() == 1 &&
             "LUT inputs should be single-bit integers");
      return rewriter
          .create<lwe::TrivialEncryptOp>(
              op->getLoc(), ctxtTy,
              lwe::EncodeOp::create(rewriter, op->getLoc(), ptxtTy, input,
                                    plaintextBits, overflow),
              ciphertextBits)
          .getResult();
    }
    return input;
  }));
}

}  // namespace

class SecretTypeConverter : public TypeConverter {
 public:
  SecretTypeConverter(MLIRContext* ctx, int minBitWidth) {
    this->minBitWidth = minBitWidth;
    // Convert secret types to LWE ciphertext types
    addConversion([](Type type) { return type; });
    addConversion([ctx, this](secret::SecretType type) -> Type {
      return getLWECiphertextForInt(ctx, type.getValueType());
    });
  }

  Type getLWECiphertextForInt(MLIRContext* ctx, Type type) const {
    if (IntegerType intType = dyn_cast<IntegerType>(type)) {
      if (intType.getWidth() == 1) {
        return lwe::getDefaultCGGICiphertextType(ctx, 1);
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

  inline Attribute defaultContext(MLIRContext* ctx) const {
    return UnitAttr::get(ctx);
  }

  int minBitWidth;
};

class SecretGenericOpLUTConversion
    : public SecretGenericOpConversion<comb::TruthTableOp, cggi::Lut3Op> {
  using SecretGenericOpConversion<comb::TruthTableOp,
                                  cggi::Lut3Op>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
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

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
    memref::LoadOp loadOp =
        cast<memref::LoadOp>(op.getBody()->getOperations().front());
    if (auto lweType = dyn_cast<lwe::LWECiphertextType>(outputTypes[0])) {
      return rewriter
          .replaceOpWithNewOp<memref::LoadOp>(op, inputs[0],
                                              loadOp.getIndices())
          .getOperation();
    }
    auto* newOp = convertReadOpInterface(loadOp, loadOp.getIndices(), inputs[0],
                                         outputTypes[0], rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

class SecretGenericOpMemRefAllocConversion
    : public SecretGenericOpConversion<memref::AllocOp> {
  using SecretGenericOpConversion<memref::AllocOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
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

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
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

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
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

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
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
    auto* newOp = convertReadOpInterface(
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

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
    affine::AffineStoreOp storeOp =
        cast<affine::AffineStoreOp>(op.getBody()->getOperations().front());
    auto toMemRef = cast<TypedValue<MemRefType>>(inputs[1]);
    auto indices = affine::expandAffineMap(
        rewriter, op.getLoc(), storeOp.getAffineMap(), storeOp.getIndices());
    if (!indices) {
      op.emitError() << "expected affine access indices";
    }
    auto* newOp = convertWriteOpInterface(
        storeOp, {indices.value().begin(), indices.value().end()}, inputs[0],
        toMemRef, rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

class SecretGenericOpMemRefStoreConversion
    : public SecretGenericOpConversion<memref::StoreOp> {
  using SecretGenericOpConversion<memref::StoreOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter& rewriter) const override {
    memref::StoreOp storeOp =
        cast<memref::StoreOp>(op.getBody()->getOperations().front());
    auto toMemRef = cast<TypedValue<MemRefType>>(inputs[1]);
    auto* newOp = convertWriteOpInterface(
        storeOp, {storeOp.getIndices().begin(), storeOp.getIndices().end()},
        inputs[0], toMemRef, rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

// ConvertTruthTableOp converts truth table ops with fully plaintext values.
struct ConvertTruthTableOp : public OpConversionPattern<comb::TruthTableOp> {
  ConvertTruthTableOp(mlir::MLIRContext* context)
      : OpConversionPattern<comb::TruthTableOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      comb::TruthTableOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op->getNumOperands() != 3) {
      op->emitError() << "expected 3 truth table arguments to lower to CGGI";
    }
    // A plaintext truth table op should not be contained inside a generic.
    assert(op->getParentOfType<secret::GenericOp>() == nullptr);

    // Create a truth table op out of arithmetic statements.
    Value t = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    Value f = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
    rewriter.replaceOp(op, buildSelectTruthTable(op.getLoc(), rewriter, t, f,
                                                 op.getLookupTable().getValue(),
                                                 adaptor.getInputs()));

    return success();
  }
};

// ConvertSecretCastOp removes secret.cast operations between multi-bit secret
// integers and tensors of single-bit secrets.
struct ConvertSecretCastOp : public OpConversionPattern<secret::CastOp> {
  ConvertSecretCastOp(mlir::MLIRContext* context)
      : OpConversionPattern<secret::CastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // The original secret cast reconciles multi-bit or memrefs of values like
    // secret<i8> to secret<memref<8xi1>> or secret<memref<2x4xi2>> and
    // secret<memref<16xi1>> and vice versa. One of these will always be a
    // flattened memref<Nxi1> type that Yosys uses as input/output types. After
    // secret-to-cggi type conversion, both will be memref types.
    auto lhsMemRefTy = dyn_cast<MemRefType>(adaptor.getInput().getType());
    auto rhsMemRefTy = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getOutput().getType()));

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
        auto castOp = memref::ReinterpretCastOp::create(
            rewriter, op.getLoc(), rhsMemRefTy, adaptor.getInput(), offset,
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
struct ConvertSecretConcealOp : public OpConversionPattern<secret::ConcealOp> {
  ConvertSecretConcealOp(mlir::MLIRContext* context)
      : OpConversionPattern<secret::ConcealOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type convertedTy =
        getTypeConverter()->convertType(op.getResult().getType());
    auto ctTy = cast<lwe::LWECiphertextType>(getElementTypeOrSelf(convertedTy));
    auto ptxtTy = lwe::LWEPlaintextType::get(
        b.getContext(), ctTy.getApplicationData(), ctTy.getPlaintextSpace());
    auto plaintextBits = rewriter.getIndexAttr(ptxtTy.getPlaintextSpace()
                                                   .getRing()
                                                   .getCoefficientType()
                                                   .getIntOrFloatBitWidth());
    auto ciphertextBits = rewriter.getIndexAttr(ctTy.getCiphertextSpace()
                                                    .getRing()
                                                    .getCoefficientType()
                                                    .getIntOrFloatBitWidth());

    auto storeElement = [&](TypedValue<IntegerType> element,
                            SmallVector<Value> indices, Value memref) {
      auto point = b.saveInsertionPoint();
      auto valueType = element.getType();
      if (valueType.getWidth() == 1) {
        auto ctValue = lwe::TrivialEncryptOp::create(
            b, ctTy,
            lwe::EncodeOp::create(b, ptxtTy, element, plaintextBits,
                                  ctTy.getApplicationData().getOverflow()),
            ciphertextBits);
        memref::StoreOp::create(b, ctValue, memref, indices);
        return;
      }
      auto loop = mlir::affine::AffineForOp::create(b, 0, valueType.getWidth());
      b.setInsertionPointToStart(loop.getBody());
      auto idx = loop.getInductionVar();

      auto one = arith::ConstantOp::create(
          b, valueType, rewriter.getIntegerAttr(valueType, 1));
      auto shiftAmount = arith::IndexCastOp::create(b, valueType, idx);
      auto bitMask = arith::ShLIOp::create(b, valueType, one, shiftAmount);
      auto andOp = arith::AndIOp::create(b, element, bitMask);
      auto shifted = arith::ShRSIOp::create(b, andOp, shiftAmount);
      auto bitValue = arith::TruncIOp::create(b, b.getI1Type(), shifted);
      auto ctValue = lwe::TrivialEncryptOp::create(
          b, ctTy,
          lwe::EncodeOp::create(b, ptxtTy, bitValue, plaintextBits,
                                ctTy.getApplicationData().getOverflow()),
          ciphertextBits);
      indices.append({idx});
      memref::StoreOp::create(b, ctValue, memref, indices);
      b.restoreInsertionPoint(point);
    };

    Value newValue;
    Value valueToStore = adaptor.getCleartext();
    if (auto memrefTy = dyn_cast<MemRefType>(convertedTy)) {
      auto allocOp = memref::AllocOp::create(b, memrefTy);
      newValue = allocOp.getResult();
      if (auto inputMemrefTy =
              dyn_cast<MemRefType>(adaptor.getCleartext().getType())) {
        // The input was a memref<<SHAPE>xiN>, so we need to extract each value
        // of the input and store it in the output memref.
        SmallVector<Value> constIndices;
        const auto* maxDimension = llvm::max_element(inputMemrefTy.getShape());
        for (auto i = 0; i < *maxDimension; ++i) {
          constIndices.push_back(arith::ConstantIndexOp::create(b, i));
        }
        for (auto i = 0; i < inputMemrefTy.getNumElements(); ++i) {
          // Extract the value from the original memref.
          auto rawIndices = unflattenIndex(i, inputMemrefTy.getShape(), 0);
          auto indices = llvm::map_to_vector(
              rawIndices,
              [&](int64_t index) -> Value { return constIndices[index]; });
          auto extractedValue =
              memref::LoadOp::create(b, valueToStore, indices).getResult();
          storeElement(cast<TypedValue<IntegerType>>(extractedValue), indices,
                       newValue);
        }
      } else {
        // The input was an iN and the output was a memref<Nxct>.
        auto singleValue = cast<TypedValue<IntegerType>>(valueToStore);
        storeElement(singleValue, {}, newValue);
      }
    } else {
      // The input was an i1 and the output was a scalar ct.
      assert(cast<IntegerType>(valueToStore.getType()).getWidth() == 1);
      newValue = lwe::TrivialEncryptOp::create(
          b, ctTy,
          lwe::EncodeOp::create(b, ptxtTy, valueToStore, plaintextBits,
                                ctTy.getApplicationData().getOverflow()),
          ciphertextBits);
    }

    rewriter.replaceAllOpUsesWith(op, newValue);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ResolveUnrealizedConversionCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  ResolveUnrealizedConversionCast(mlir::MLIRContext* context)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter& rewriter) const override {
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

int findLUTSize(MLIRContext* context, Operation* module) {
  int maxIntSize = 1;
  auto processOperation = [&](Operation* op) {
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
    MLIRContext* context = &getContext();
    auto* module = getOperation();

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

    patterns
        .add<SecretGenericOpLUTConversion, SecretGenericOpMemRefAllocConversion,
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
             ConvertSecretConcealOp, ConvertAny<affine::AffineForOp>,
             ConvertAny<affine::AffineYieldOp>>(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
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
