#include "include/Conversion/CombToCGGI/CombToCGGI.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "include/Dialect/CGGI/IR/CGGIOps.h"
#include "include/Dialect/Comb/IR/CombOps.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Dialect/LWE/IR/LWEOps.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Conversion/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"         // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::comb {

#define GEN_PASS_DEF_COMBTOCGGI
#include "include/Conversion/CombToCGGI/CombToCGGI.h.inc"

namespace {

// buildSelectTruthTable recursively creates arithmetic operations to compute a
// lookup table on plaintext integers.
Value buildSelectTruthTable(Location loc, OpBuilder &b, Value t, Value f,
                            const APInt &lut, ValueRange lutInputs) {
  int tableSize = lut.getBitWidth();
  assert(tableSize == (1ull << lutInputs.size()));
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

// equivalentMultiBitAndMemRefchecks whether the candidateMultiBit integer type
// is equivalent to the candidateMemRef type.
// Return true if the candidateMemRef is a memref of single bits with
// size equal to the number of bits of the candidateMultiBit.
bool equivalentMultiBitAndMemRef(Type candidateMultiBit, Type candidateMemRef) {
  if (auto multiBitTy = dyn_cast<IntegerType>(candidateMultiBit)) {
    if (auto memrefTy = dyn_cast<MemRefType>(candidateMemRef)) {
      auto eltTy = dyn_cast<IntegerType>(memrefTy.getElementType());
      if (eltTy && multiBitTy.getWidth() ==
                       memrefTy.getNumElements() * eltTy.getWidth()) {
        return true;
      }
    }
  }
  return false;
}

Operation *convertWriteOpInterface(Operation *op, SmallVector<Value> indices,
                                   Value valueToStore,
                                   TypedValue<MemRefType> toMemRef,
                                   ConversionPatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  MemRefType toMemRefTy = toMemRef.getType();
  Type valueToStoreType = valueToStore.getType();
  return llvm::TypeSwitch<Type, Operation *>(valueToStoreType)
      // Plaintext integer into memref
      .Case<IntegerType>([&](auto valType) {
        auto ctTy = toMemRefTy.getElementType();
        auto encoding = cast<lwe::LWECiphertextType>(ctTy).getEncoding();
        auto ptxtTy = lwe::LWEPlaintextType::get(b.getContext(), encoding);

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

Operation *convertReadOpInterface(Operation *op, SmallVector<Value> indices,
                                  Value fromMemRef, Type outputType,
                                  ConversionPatternRewriter &rewriter) {
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

SmallVector<Value> encodeInputs(Operation *op, ValueRange inputs,
                                ConversionPatternRewriter &rewriter) {
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

class SecretTypeConverter : public TypeConverter {
 public:
  SecretTypeConverter(MLIRContext *ctx, int minBitWidth)
      : minBitWidth(minBitWidth) {
    addConversion([](Type type) { return type; });

    // Convert secret types to LWE ciphertext types
    addConversion([ctx, this](secret::SecretType type) -> Type {
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

  int minBitWidth;
};

template <typename T>
class SecretGenericOpConversion
    : public OpConversionPattern<secret::GenericOp> {
 public:
  using OpConversionPattern<secret::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (op.getBody()->getOperations().size() > 2) {
      // Each secret.generic should contain at most one instruction -
      // secret-distribute-generic can be used to distribute through the
      // combinational ops.
      return failure();
    }

    auto &innerOp = op.getBody()->getOperations().front();
    if (!isa<T>(innerOp)) {
      return failure();
    }

    // Assemble the arguments for the CGGI operation.
    SmallVector<Value> inputs;
    for (OpOperand &operand : innerOp.getOpOperands()) {
      if (auto *secretArg = op.getOpOperandForBlockArgument(operand.get())) {
        inputs.push_back(
            adaptor.getODSOperands(0)[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand.get());
      }
    }

    // Convert the secret result types to ciphertext types.
    SmallVector<Type> outputTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), outputTypes))) {
      return failure();
    }

    static_cast<const SecretGenericOpConversion<T> *>(this)->replaceOp(
        op, outputTypes, inputs, innerOp.getAttrs(), rewriter);
    return success();
  }

  // Function to replacing an combinational operation T with a CGGI equivalent
  // operation Y.
  virtual void replaceOp(secret::GenericOp op, TypeRange outputTypes,
                         ValueRange inputs, ArrayRef<NamedAttribute> attributes,
                         ConversionPatternRewriter &rewriter) const = 0;
};

class SecretGenericOpLUTConversion
    : public SecretGenericOpConversion<comb::TruthTableOp> {
  using SecretGenericOpConversion<
      comb::TruthTableOp>::SecretGenericOpConversion;

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ArrayRef<NamedAttribute> attributes,
                 ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> encodedInputs =
        encodeInputs(op.getOperation(), inputs, rewriter);

    // Assemble the lookup table.
    comb::TruthTableOp truthOp =
        cast<comb::TruthTableOp>(op.getBody()->getOperations().front());
    rewriter.replaceOpWithNewOp<cggi::Lut3Op>(
        op, encodedInputs[0], encodedInputs[1], encodedInputs[2],
        truthOp.getLookupTable());
  }
};

class SecretGenericOpMemRefLoadConversion
    : public SecretGenericOpConversion<memref::LoadOp> {
  using SecretGenericOpConversion<memref::LoadOp>::SecretGenericOpConversion;

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ArrayRef<NamedAttribute> attributes,
                 ConversionPatternRewriter &rewriter) const override {
    memref::LoadOp loadOp =
        cast<memref::LoadOp>(op.getBody()->getOperations().front());
    if (auto lweType = dyn_cast<lwe::LWECiphertextType>(outputTypes[0])) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, inputs[0],
                                                  loadOp.getIndices());
      return;
    }
    rewriter.replaceOp(
        op, convertReadOpInterface(loadOp, loadOp.getIndices(), inputs[0],
                                   outputTypes[0], rewriter));
  }
};

template <typename GateOp, typename CGGIGateOp>
class SecretGenericOpGateConversion : public SecretGenericOpConversion<GateOp> {
  using SecretGenericOpConversion<GateOp>::SecretGenericOpConversion;

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ArrayRef<NamedAttribute> attributes,
                 ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<CGGIGateOp>(
        op, outputTypes, encodeInputs(op.getOperation(), inputs, rewriter),
        attributes);
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

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ArrayRef<NamedAttribute> attributes,
                 ConversionPatternRewriter &rewriter) const override {
    affine::AffineLoadOp loadOp =
        cast<affine::AffineLoadOp>(op.getBody()->getOperations().front());
    if (auto lweType = dyn_cast<lwe::LWECiphertextType>(outputTypes[0])) {
      rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
          op, inputs[0], loadOp.getAffineMap(), loadOp.getIndices());
      return;
    }
    auto indices = affine::expandAffineMap(
        rewriter, op.getLoc(), loadOp.getAffineMap(), loadOp.getIndices());
    if (!indices) {
      op.emitError() << "expected affine access indices";
    }
    rewriter.replaceOp(
        op, convertReadOpInterface(
                loadOp, {indices.value().begin(), indices.value().end()},
                inputs[0], outputTypes[0], rewriter));
  }
};

class SecretGenericOpAffineStoreConversion
    : public SecretGenericOpConversion<affine::AffineStoreOp> {
  using SecretGenericOpConversion<
      affine::AffineStoreOp>::SecretGenericOpConversion;

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ArrayRef<NamedAttribute> attributes,
                 ConversionPatternRewriter &rewriter) const override {
    affine::AffineStoreOp storeOp =
        cast<affine::AffineStoreOp>(op.getBody()->getOperations().front());
    auto toMemRef = cast<TypedValue<MemRefType>>(inputs[1]);
    if (auto lweType = dyn_cast<lwe::LWECiphertextType>(inputs[0].getType())) {
      rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
          op, inputs[0], toMemRef, storeOp.getAffineMap(),
          storeOp.getIndices());
      return;
    }
    auto indices = affine::expandAffineMap(
        rewriter, op.getLoc(), storeOp.getAffineMap(), storeOp.getIndices());
    if (!indices) {
      op.emitError() << "expected affine access indices";
    }
    rewriter.replaceOp(
        op, convertWriteOpInterface(
                storeOp, {indices.value().begin(), indices.value().end()},
                inputs[0], toMemRef, rewriter));
  }
};

class SecretGenericOpMemRefStoreConversion
    : public SecretGenericOpConversion<memref::StoreOp> {
  using SecretGenericOpConversion<memref::StoreOp>::SecretGenericOpConversion;

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ArrayRef<NamedAttribute> attributes,
                 ConversionPatternRewriter &rewriter) const override {
    memref::StoreOp storeOp =
        cast<memref::StoreOp>(op.getBody()->getOperations().front());
    auto toMemRef = cast<TypedValue<MemRefType>>(inputs[1]);
    if (auto lweType = dyn_cast<lwe::LWECiphertextType>(inputs[0].getType())) {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(op, inputs[0], toMemRef,
                                                   storeOp.getIndices());
      return;
    }
    rewriter.replaceOp(
        op,
        convertWriteOpInterface(
            storeOp, {storeOp.getIndices().begin(), storeOp.getIndices().end()},
            inputs[0], toMemRef, rewriter));
  }
};

class SecretGenericOpMemRefAllocConversion
    : public SecretGenericOpConversion<memref::AllocOp> {
  using SecretGenericOpConversion<memref::AllocOp>::SecretGenericOpConversion;

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ArrayRef<NamedAttribute> attributes,
                 ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, outputTypes, inputs,
                                                 attributes);
  }
};

class SecretGenericOpMemRefDeallocConversion
    : public SecretGenericOpConversion<memref::DeallocOp> {
  using SecretGenericOpConversion<memref::DeallocOp>::SecretGenericOpConversion;

  void replaceOp(secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
                 ArrayRef<NamedAttribute> attributes,
                 ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, outputTypes, inputs,
                                                   attributes);
  }
};

// ConvertTruthTableOp converts truth table ops with fully plaintext values.
struct ConvertTruthTableOp : public OpConversionPattern<TruthTableOp> {
  ConvertTruthTableOp(mlir::MLIRContext *context)
      : OpConversionPattern<TruthTableOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TruthTableOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
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
struct ConvertSecretCastOp : public OpConversionPattern<secret::CastOp> {
  ConvertSecretCastOp(mlir::MLIRContext *context)
      : OpConversionPattern<secret::CastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // If this is a cast from secret<i8> to secret<tensor<8xi1>> or vice
    // versa, replace with the cast's input.
    auto inputTy =
        cast<secret::SecretType>(op.getInput().getType()).getValueType();
    auto outputTy =
        cast<secret::SecretType>(op.getOutput().getType()).getValueType();

    if (equivalentMultiBitAndMemRef(inputTy, outputTy) ||
        equivalentMultiBitAndMemRef(outputTy, inputTy)) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }
    return failure();
  }
};

struct CombToCGGI : public impl::CombToCGGIBase<CombToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    // TODO(#250): The lutSize here is fixed on the assumption that the comb
    // dialect is using ternary LUTs. Generalize lutSize by doing an analysis
    // pass on the input combinational operations and integers.
    int lutSize = 3;
    SecretTypeConverter typeConverter(context, lutSize);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    patterns
        .add<SecretGenericOpLUTConversion, SecretGenericOpMemRefAllocConversion,
             SecretGenericOpMemRefDeallocConversion,
             SecretGenericOpMemRefLoadConversion,
             SecretGenericOpAffineStoreConversion,
             SecretGenericOpAffineLoadConversion,
             SecretGenericOpMemRefStoreConversion, ConvertTruthTableOp,
             SecretGenericOpInvConversion, SecretGenericOpAndConversion,
             SecretGenericOpNorConversion, SecretGenericOpNandConversion,
             SecretGenericOpOrConversion, SecretGenericOpXNorConversion,
             SecretGenericOpXorConversion, ConvertSecretCastOp>(typeConverter,
                                                                context);
    target.addIllegalOp<TruthTableOp, secret::CastOp, secret::GenericOp>();
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      // Legal only when the memref element type matches the stored type.
      return op.getMemRefType().getElementType() ==
             op.getValueToStore().getType();
    });
    target.addDynamicallyLegalOp<affine::AffineStoreOp>(
        [&](affine::AffineStoreOp op) {
          // Legal only when the memref element type matches the stored type.
          return op.getMemRefType().getElementType() ==
                 op.getValueToStore().getType();
        });
    target.addDynamicallyLegalOp<memref::LoadOp>([&](memref::LoadOp op) {
      // Legal only when the memref element type matches the loaded type.
      return op.getMemRefType().getElementType() == op.getResult().getType();
    });
    target.addDynamicallyLegalOp<affine::AffineLoadOp>(
        [&](affine::AffineLoadOp op) {
          // Legal only when the memref element type matches the loaded type.
          return op.getMemRefType().getElementType() ==
                 op.getResult().getType();
        });
    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::comb
