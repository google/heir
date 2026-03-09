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
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
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
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
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

Operation* convertWriteOpInterface(
    Operation* op, SmallVector<Value> indices, Value valueToStore,
    TypedValue<TensorType> toTensor,
    ContextAwareConversionPatternRewriter& rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  TensorType toTensorTy = toTensor.getType();
  Type valueToStoreType = valueToStore.getType();
  return llvm::TypeSwitch<Type, Operation*>(valueToStoreType)
      // Plaintext integer into tensor
      .Case<IntegerType>([&](auto valType) {
        auto ctTy =
            dyn_cast<lwe::LWECiphertextType>(toTensorTy.getElementType());
        auto ptxtTy = lwe::LWEPlaintextType::get(b.getContext(),
                                                 ctTy.getPlaintextSpace());

        auto ciphertextBits = ctTy.getCiphertextSpace()
                                  .getRing()
                                  .getCoefficientType()
                                  .getIntOrFloatBitWidth();
        auto plaintextBits = ctTy.getPlaintextSpace()
                                 .getRing()
                                 .getCoefficientType()
                                 .getIntOrFloatBitWidth();

        if (valType.getWidth() == 1) {
          auto ctValue = lwe::TrivialEncryptOp::create(
              b, ctTy,
              lwe::EncodeOp::create(b, op->getLoc(), ptxtTy, valueToStore,
                                    b.getIndexAttr(plaintextBits)),
              b.getIndexAttr(ciphertextBits));

          return tensor::InsertOp::create(b, ctValue, toTensor, indices);
        }

        // Get i-th bit of input and insert the bit into the tensor of
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
            lwe::EncodeOp::create(b, op->getLoc(), ptxtTy, bitValue,
                                  b.getIndexAttr(plaintextBits)),
            b.getIndexAttr(ciphertextBits));

        indices.push_back(idx);
        return tensor::InsertOp::create(b, ctValue, toTensor, indices);
      })
      .Case<lwe::LWECiphertextType>([&](auto valType) {
        return tensor::InsertOp::create(b, valueToStore, toTensor, indices);
      })
      .Case<TensorType>([&](TensorType valType) {
        int rank = toTensorTy.getRank();

        // A store op with a tensor value to store must have
        // originated from a secret encoding a multi-bit value. Under type
        // conversion, the op is storing a tensor<BITSIZE!ct_ty> into a
        // tensor of the form tensor<?xBITSIZE!ct_ty>. Use an insert slice:
        //   * Offset: offset at the storage index [indices, 0]
        //   * Strides: match original tensor [1, 1, ..., 1]
        //   * Sizes: rank-reduce to the last dim [1, 1, ..., BITSIZE]
        SmallVector<OpFoldResult> offsets = getAsOpFoldResult(indices);
        offsets.push_back(OpFoldResult(b.getIndexAttr(0)));

        Attribute oneIdxAttr = rewriter.getIndexAttr(1);
        SmallVector<OpFoldResult> strides(rank, oneIdxAttr);
        SmallVector<OpFoldResult> sizes(rank - 1, oneIdxAttr);
        sizes.push_back(rewriter.getIndexAttr(toTensorTy.getShape()[rank - 1]));
        return tensor::InsertSliceOp::create(b, valueToStore, toTensor, offsets,
                                             sizes, strides);
      });
  llvm_unreachable("expected integer or tensor to store in ciphertext tensor");
}

Operation* convertReadOpInterface(
    Operation* op, SmallVector<Value> indices, Value fromTensor,
    Type outputType, ContextAwareConversionPatternRewriter& rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  RankedTensorType outputTy = cast<RankedTensorType>(outputType);
  TensorType fromTensorType = cast<TensorType>(fromTensor.getType());
  int rank = fromTensorType.getRank();

  // A load op with a tensor value to store must have
  // originated from a secret encoding a multi-bit value. Under type
  // conversion, the op is loading a tensor<BITSIZE!ct_ty> from a
  // tensor<?xBITSIZE!ct_ty>.

  // Extract a slice from the tensor:
  //   * Offset: offset at the storage index [indices, 0]
  //   * Strides: match original tensor [1, 1, ..., 1]
  //   * Sizes: rank-reduce to the last dim [1, 1, ..., BITSIZE]
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets = getAsOpFoldResult(indices);
  offsets.push_back(OpFoldResult(b.getIndexAttr(0)));
  SmallVector<OpFoldResult> strides(rank, oneIdxAttr);
  SmallVector<OpFoldResult> sizes(rank - 1, oneIdxAttr);
  sizes.push_back(rewriter.getIndexAttr(fromTensorType.getShape()[rank - 1]));

  return tensor::ExtractSliceOp::create(b, outputTy, fromTensor, offsets, sizes,
                                        strides);
}

SmallVector<Value> encodeInputs(
    Operation* op, ValueRange inputs,
    ContextAwareConversionPatternRewriter& rewriter) {
  // Get the ciphertext type.
  lwe::LWECiphertextType ctxtTy;
  for (auto input : inputs) {
    if (isa<lwe::LWECiphertextType>(input.getType())) {
      ctxtTy = cast<lwe::LWECiphertextType>(input.getType());
      break;
    }
  }

  // Encode any plaintexts in the inputs.
  auto ptxtSpace = ctxtTy.getPlaintextSpace();
  auto plaintextBits = rewriter.getIndexAttr(
      ptxtSpace.getRing().getCoefficientType().getIntOrFloatBitWidth());
  auto ciphertextBits = rewriter.getIndexAttr(ctxtTy.getCiphertextSpace()
                                                  .getRing()
                                                  .getCoefficientType()
                                                  .getIntOrFloatBitWidth());
  auto ptxtTy = lwe::LWEPlaintextType::get(rewriter.getContext(), ptxtSpace);
  return llvm::to_vector(llvm::map_range(inputs, [&](auto input) -> Value {
    if (!isa<lwe::LWECiphertextType>(input.getType())) {
      IntegerType integerTy = dyn_cast<IntegerType>(input.getType());
      assert(integerTy && integerTy.getWidth() == 1 &&
             "LUT inputs should be single-bit integers");
      return lwe::TrivialEncryptOp::create(
                 rewriter, op->getLoc(), ctxtTy,
                 lwe::EncodeOp::create(rewriter, op->getLoc(), ptxtTy, input,
                                       plaintextBits),
                 ciphertextBits)
          .getResult();
    }
    return input;
  }));
}

}  // namespace

class SecretTypeConverter : public ContextAwareTypeConverter {
 public:
  SecretTypeConverter(MLIRContext* ctx, int minBitWidth) {
    this->minBitWidth = minBitWidth;
    // Convert secret types to LWE ciphertext types
    addConversion([](Type type, Attribute attr) { return type; });
    addConversion([ctx, this](secret::SecretType type, Attribute attr) -> Type {
      return getLWECiphertextForInt(ctx, type.getValueType());
    });
  }

  Type getLWECiphertextForInt(MLIRContext* ctx, Type type) const {
    if (IntegerType intType = dyn_cast<IntegerType>(type)) {
      if (intType.getWidth() == 1) {
        return lwe::getDefaultCGGICiphertextType(ctx, this->minBitWidth);
      }
      return RankedTensorType::get(
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
      return RankedTensorType::get(newShape, elementShape.getElementType());
    }
    return shapedType.cloneWith(newShape, elementType);
  }

  inline Attribute defaultContext(MLIRContext* ctx) const {
    return UnitAttr::get(ctx);
  }

  FailureOr<Attribute> getContextualAttr(Value value) const override {
    // No attribute is necessary yet, since we aren't attaching contextual
    // information to the types yet.
    return defaultContext(value.getContext());
  }

  LogicalResult convertFuncSignature(
      FunctionOpInterface funcOp, SmallVectorImpl<Type>& newArgTypes,
      SmallVectorImpl<Type>& newResultTypes) const override {
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

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    SmallVector<Value> encodedInputs =
        encodeInputs(op.getOperation(), inputs, rewriter);

    // Assemble the lookup table.
    comb::TruthTableOp truthOp =
        cast<comb::TruthTableOp>(op.getBody()->getOperations().front());

    if (encodedInputs.size() == 3)
      return rewriter
          .replaceOpWithNewOp<cggi::Lut3Op>(op, encodedInputs[0],
                                            encodedInputs[1], encodedInputs[2],
                                            truthOp.getLookupTable())
          .getOperation();
    if (encodedInputs.size() == 4)
      return rewriter
          .replaceOpWithNewOp<cggi::Lut4Op>(
              op, encodedInputs[0], encodedInputs[1], encodedInputs[2],
              encodedInputs[3], truthOp.getLookupTable())
          .getOperation();
    return rewriter.notifyMatchFailure(op, "expected 3 or 4 LUT inputs");
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
      ContextAwareConversionPatternRewriter& rewriter) const override {
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

class SecretGenericOpTensorExtractConversion
    : public SecretGenericOpConversion<tensor::ExtractOp> {
  using SecretGenericOpConversion<tensor::ExtractOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    tensor::ExtractOp extractOp =
        cast<tensor::ExtractOp>(op.getBody()->getOperations().front());
    if (auto lweType = dyn_cast<lwe::LWECiphertextType>(outputTypes[0])) {
      return rewriter
          .replaceOpWithNewOp<tensor::ExtractOp>(op, inputs[0],
                                                 extractOp.getIndices())
          .getOperation();
    }
    auto* newOp = convertReadOpInterface(extractOp, extractOp.getIndices(),
                                         inputs[0], outputTypes[0], rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};  // namespace mlir::heir

class SecretGenericOpTensorInsertConversion
    : public SecretGenericOpConversion<tensor::InsertOp> {
  using SecretGenericOpConversion<tensor::InsertOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    tensor::InsertOp insertOp =
        cast<tensor::InsertOp>(op.getBody()->getOperations().front());
    auto toTensor = cast<TypedValue<TensorType>>(inputs[1]);
    if (!isa<lwe::LWECiphertextType>(toTensor.getType().getElementType())) {
      // We may be inserting into a tensor initialized with plaintexts. The
      // pattern ConvertPlaintextTensorInsertOp should apply first.
      return rewriter.notifyMatchFailure(
          op, "tensor element type is not a ciphertext");
    }
    auto* newOp = convertWriteOpInterface(
        insertOp, {insertOp.getIndices().begin(), insertOp.getIndices().end()},
        inputs[0], toTensor, rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

class SecretGenericOpTensorInsertSliceConversion
    : public SecretGenericOpConversion<tensor::InsertSliceOp> {
  using SecretGenericOpConversion<
      tensor::InsertSliceOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    tensor::InsertSliceOp insertOp =
        cast<tensor::InsertSliceOp>(op.getBody()->getOperations().front());
    auto toTensor = cast<TypedValue<TensorType>>(inputs[1]);
    auto* newOp = convertWriteOpInterface(
        insertOp, {insertOp.getOffsets().begin(), insertOp.getOffsets().end()},
        inputs[0], toTensor, rewriter);
    rewriter.replaceOp(op, newOp);
    return newOp;
  }
};

// ConvertTruthTableOp converts truth table ops with fully plaintext values.
struct ConvertTruthTableOp
    : public ContextAwareOpConversionPattern<comb::TruthTableOp> {
  ConvertTruthTableOp(mlir::MLIRContext* context)
      : ContextAwareOpConversionPattern<comb::TruthTableOp>(context) {}

  using ContextAwareOpConversionPattern::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      comb::TruthTableOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const override {
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
struct ConvertSecretCastOp
    : public ContextAwareOpConversionPattern<secret::CastOp> {
  ConvertSecretCastOp(mlir::MLIRContext* context)
      : ContextAwareOpConversionPattern<secret::CastOp>(context) {}

  using ContextAwareOpConversionPattern::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::CastOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    // The original secret cast reconciles multi-bit or tensors of values like
    // secret<i8> to secret<tensor<8xi1>> or secret<tensor<2x4xi2>> and
    // secret<tensor<16xi1>> and vice versa. One of these will always be a
    // flattened tensor<Nxi1> type that Yosys uses as input/output types.
    // After secret-to-cggi type conversion, both will be tensor types.
    auto lhsTensorTy = dyn_cast<TensorType>(adaptor.getInput().getType());
    auto rhsTensorTy = dyn_cast<TensorType>(typeConverter->convertType(
        op.getOutput().getType(),
        typeConverter->getContextualAttr(op.getOutput()).value_or(nullptr)));

    if (!lhsTensorTy || !rhsTensorTy) {
      return op->emitOpError()
             << "expected cast between multi-bit secret types, got "
             << op.getInput().getType() << " and " << op.getOutput().getType();
    }

    int lhsBits = lhsTensorTy.getNumElements();
    int rhsBits = rhsTensorTy.getNumElements();
    if (lhsBits != rhsBits) {
      return op->emitOpError()
             << "expected cast between secrets holding the "
                "same number of total bits, got "
             << op.getInput().getType() << " and " << op.getOutput().getType();
    }

    // Fold the cast if the resulting type converted shapes are the same.
    if (lhsTensorTy.getShape() == rhsTensorTy.getShape()) {
      // This happens when one of the original shapes was a tensor and the other
      // was an integer, for e.g. a secret.cast from i8 to tensor<8xlwe_ct>.
      assert(isa<ShapedType>(op.getInput().getType().getValueType()) !=
             isa<ShapedType>(op.getOutput().getType().getValueType()));
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Otherwise, resolve the cast by reshaping the input. This happens when the
    // original operand or result was a tensor which Yosys optimizer flattened,
    // for e.g. secret<tensor<2xi4>> (tensor<8xlwe_ct>).
    if (lhsTensorTy && rhsTensorTy) {
      if (lhsTensorTy.getRank() > rhsTensorTy.getRank() &&
          rhsTensorTy.getRank() == 1) {
        // This case happens when converting a high dimension tensor into a flat
        // tensor of single bits as an operand to a Yosys optimized body, e.g.
        // secret<tensor<1x1xi8>> (tensor<1x1x8xlwe_ciphertext>) to
        // secret<tensor<8xi1>> (tensor<8xlwe_ciphertext>). In this case,
        // collapse the tensor shape.
        SmallVector<mlir::ReassociationIndices> reassociation;
        auto range = llvm::seq<unsigned>(0, lhsTensorTy.getRank());
        reassociation.emplace_back(range.begin(), range.end());
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, adaptor.getInput(), reassociation);
        return success();
      } else if (lhsTensorTy.getRank() < rhsTensorTy.getRank() &&
                 lhsTensorTy.getRank() == 1) {
        // This is the case of converting results of Yosys optimized bodies,
        // flat tensors of single bits, into a high dimensional tensor, e.g.
        // secret<tensor<8xi1>> (tensor<8xlwe_ciphertext>) to
        // secret<tensor<1x2xi4>> (tensor<1x2x4xlwe_ciphertext>).
        SmallVector<mlir::ReassociationIndices> reassociation;
        auto range = llvm::seq<unsigned>(0, rhsTensorTy.getRank());
        reassociation.emplace_back(range.begin(), range.end());
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, rhsTensorTy, adaptor.getInput(), reassociation);
        return success();
      } else if (lhsTensorTy.getShape() != rhsTensorTy.getShape()) {
        // Fallback to a reshape to resolve the tensor shapes.
        auto shapeOp = mlir::arith::ConstantOp::create(
            rewriter, op.getLoc(),
            RankedTensorType::get(rhsTensorTy.getShape().size(),
                                  rewriter.getIndexType()),
            rewriter.getIndexTensorAttr(rhsTensorTy.getShape()));
        auto castOp = tensor::ReshapeOp::create(
            rewriter, op.getLoc(), rhsTensorTy, adaptor.getInput(), shapeOp);
        rewriter.replaceOp(op, castOp);
        return success();
      }
    }

    return rewriter.notifyMatchFailure(op, "unsupported operand types");
  }
};

// ConvertSecretConcealOp lowers secret.conceal to a series of trivial_encrypt
// ops stored into a tensor.
struct ConvertSecretConcealOp
    : public ContextAwareOpConversionPattern<secret::ConcealOp> {
  ConvertSecretConcealOp(mlir::MLIRContext* context)
      : ContextAwareOpConversionPattern<secret::ConcealOp>(context) {}

  using ContextAwareOpConversionPattern::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type convertedTy = getTypeConverter()->convertType(
        op.getResult().getType(), getTypeConverter()
                                      ->getContextualAttr(op.getResult())
                                      .value_or(nullptr));
    auto ctTy = cast<lwe::LWECiphertextType>(getElementTypeOrSelf(convertedTy));
    auto ptxtTy =
        lwe::LWEPlaintextType::get(b.getContext(), ctTy.getPlaintextSpace());

    Value valueToStore = adaptor.getCleartext();
    Value resultValue;

    auto encodeElement =
        [&](TypedValue<IntegerType> value) -> SmallVector<Value> {
      auto valueType = value.getType();
      if (valueType.getWidth() == 1) {
        auto plaintextBits = ctTy.getPlaintextSpace()
                                 .getRing()
                                 .getCoefficientType()
                                 .getIntOrFloatBitWidth();
        auto ciphertextBits = ctTy.getCiphertextSpace()
                                  .getRing()
                                  .getCoefficientType()
                                  .getIntOrFloatBitWidth();

        return {lwe::TrivialEncryptOp::create(
            b, b.getLoc(), ctTy,
            lwe::EncodeOp::create(b, b.getLoc(), ptxtTy, value,
                                  b.getIndexAttr(plaintextBits)),
            b.getIndexAttr(ciphertextBits))};
      }
      SmallVector<Value> elementValues;
      for (auto i = 0; i < valueType.getWidth(); ++i) {
        auto one = arith::ConstantOp::create(
            b, valueType, rewriter.getIntegerAttr(valueType, 1));
        auto shiftAmount = arith::ConstantOp::create(
            b, valueType, b.getIntegerAttr(valueType, i));
        auto bitMask = arith::ShLIOp::create(b, valueType, one, shiftAmount);
        auto andOp = arith::AndIOp::create(b, value, bitMask);
        auto shifted = arith::ShRSIOp::create(b, andOp, shiftAmount);
        auto bitValue = arith::TruncIOp::create(b, b.getI1Type(), shifted);

        auto plaintextBits = ctTy.getPlaintextSpace()
                                 .getRing()
                                 .getCoefficientType()
                                 .getIntOrFloatBitWidth();
        auto ciphertextBits = ctTy.getCiphertextSpace()
                                  .getRing()
                                  .getCoefficientType()
                                  .getIntOrFloatBitWidth();

        auto ctValue = lwe::TrivialEncryptOp::create(
            b, b.getLoc(), ctTy,
            lwe::EncodeOp::create(b, b.getLoc(), ptxtTy, bitValue,
                                  b.getIndexAttr(plaintextBits)),
            b.getIndexAttr(ciphertextBits));

        elementValues.push_back(ctValue);
      }
      return elementValues;
    };

    // The resulting tensor will be tensor<<SHAPE>xMxlwe_ct> where <SHAPE> is
    // the shape of the original tensor and M is the bit width of the elements.
    if (auto tensorTy = dyn_cast<TensorType>(convertedTy)) {
      SmallVector<Value> elementValues;
      if (auto inputTensorTy = dyn_cast<TensorType>(valueToStore.getType())) {
        SmallVector<Value> constIndices;
        const auto* maxDimension = llvm::max_element(inputTensorTy.getShape());
        for (auto i = 0; i < *maxDimension; ++i) {
          constIndices.push_back(arith::ConstantIndexOp::create(b, i));
        }
        for (auto i = 0; i < inputTensorTy.getNumElements(); ++i) {
          // Extract the value from the original tensor.
          auto rawIndices = unflattenIndex(i, inputTensorTy.getShape(), 0);
          auto indices = llvm::map_to_vector(
              rawIndices,
              [&](int64_t index) -> Value { return constIndices[index]; });
          auto extractedValue =
              tensor::ExtractOp::create(b, valueToStore, indices).getResult();
          auto newValues =
              encodeElement(cast<TypedValue<IntegerType>>(extractedValue));
          elementValues.append(newValues.begin(), newValues.end());
        }
      } else {
        auto singleValue = cast<TypedValue<IntegerType>>(valueToStore);
        elementValues = encodeElement(singleValue);
      }
      assert(elementValues.size() == tensorTy.getNumElements());
      resultValue = tensor::FromElementsOp::create(b, tensorTy, elementValues);
    } else {
      auto typedValue = dyn_cast<TypedValue<IntegerType>>(valueToStore);
      auto resultValues = encodeElement(typedValue);
      assert(resultValues.size() == 1 && "expected one encoded value");
      resultValue = resultValues[0];
    }
    rewriter.replaceOp(op, resultValue);
    return success();
  }
};

// ConvertFromElementsOp converts a tensor::FromElementsOp of plaintext/secret
// types into a tensor::FromElementsOp of lwe::LWECiphertextType of ciphertext
// types.
struct ConvertFromElementsOp
    : public SecretGenericOpConversion<tensor::FromElementsOp> {
  ConvertFromElementsOp(mlir::MLIRContext* context)
      : SecretGenericOpConversion<tensor::FromElementsOp>(context) {}
  using SecretGenericOpConversion<
      tensor::FromElementsOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensorTy = cast<TensorType>(outputTypes[0]);
    auto ctTy = dyn_cast<lwe::LWECiphertextType>(tensorTy.getElementType());
    if (!ctTy) {
      return rewriter.notifyMatchFailure(
          op, "tensor element type must be a ciphertext");
    }

    auto ptTy =
        lwe::LWEPlaintextType::get(b.getContext(), ctTy.getPlaintextSpace());
    // All elements of the operation must have the same type. If they are
    // scalars, construct a from_elements op, encoding plaintexts as necessary.
    if (!isa<ShapedType>(inputs[0].getType())) {
      SmallVector<Value> values;
      for (auto element : inputs) {
        if (isa<lwe::LWECiphertextType>(element.getType())) {
          values.push_back(element);
        } else {
          auto plaintextBits = ctTy.getPlaintextSpace()
                                   .getRing()
                                   .getCoefficientType()
                                   .getIntOrFloatBitWidth();
          auto ciphertextBits = ctTy.getCiphertextSpace()
                                    .getRing()
                                    .getCoefficientType()
                                    .getIntOrFloatBitWidth();

          auto ctElement = lwe::TrivialEncryptOp::create(
              b, b.getLoc(), ctTy,
              lwe::EncodeOp::create(b, b.getLoc(), ptTy, element,
                                    b.getIndexAttr(plaintextBits)),
              b.getIndexAttr(ciphertextBits));

          values.push_back(ctElement);
        }
      }
      return rewriter
          .replaceOpWithNewOp<tensor::FromElementsOp>(op, outputTypes[0],
                                                      values)
          .getOperation();
    }
    auto inputTensorTy = cast<TensorType>(inputs[0].getType());
    assert(inputTensorTy && "expected scalar or tensor input");
    if (tensorTy.getNumElements() == inputTensorTy.getNumElements()) {
      // We need to reshape the input tensor, which may be a tensor<32xlwe_ct>
      // to a tensor<1x1x32xlwe_ct>.
      auto shapeOp = mlir::arith::ConstantOp::create(
          b,
          RankedTensorType::get(tensorTy.getShape().size(),
                                rewriter.getIndexType()),
          rewriter.getIndexTensorAttr(tensorTy.getShape()));
      return rewriter
          .replaceOpWithNewOp<tensor::ReshapeOp>(op, tensorTy, inputs[0],
                                                 shapeOp)
          .getOperation();
    }
    // Otherwise, there are many tensor operands that will need to be
    // concatenated.
    return rewriter.notifyMatchFailure(op, "too many tensor operands");
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

static int findLUTSize(MLIRContext* context, Operation* module) {
  int maxIntSize = 1;
  auto processOperation = [&](Operation* op) {
    if (isa<comb::CombDialect>(op->getDialect())) {
      int currentSize = 0;
      if (auto ttOp = dyn_cast<comb::TruthTableOp>(op))
        currentSize = ttOp.getInputs().size();
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
                           lwe::LWEDialect, tensor::TensorDialect>();
    target.addIllegalOp<comb::TruthTableOp, secret::CastOp, secret::GenericOp,
                        secret::ConcealOp>();

    target.addDynamicallyLegalOp<tensor::FromElementsOp>(
        [&](tensor::FromElementsOp op) {
          // Legal only when the tensor element type matches the stored
          // type.
          for (auto element : op.getElements()) {
            if (element.getType() != op.getType().getElementType()) {
              return false;
            }
          }
          return true;
        });
    target.addDynamicallyLegalOp<tensor::ExtractOp>([&](tensor::ExtractOp op) {
      // Legal only when the tensor element type matches the extracted type.
      return op.getType() == op.getTensor().getType().getElementType();
    });
    target.addDynamicallyLegalOp<tensor::InsertOp>([&](tensor::InsertOp op) {
      // Legal only when the tensor element type matches the inserted
      // type.
      return op.getDest().getType().getElementType() ==
                 op.getScalar().getType() &&
             op.getType().getElementType() == op.getScalar().getType();
    });
    target.addDynamicallyLegalOp<tensor::InsertSliceOp>(
        [&](tensor::InsertSliceOp op) {
          // Legal only when the tensor element type matches the inserted
          // type.
          assert(op.getDest());
          assert(op.getSource());
          return op.getDestType().getElementType() ==
                 op.getSourceType().getElementType();
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
        SecretGenericOpLUTConversion, SecretGenericOpTensorInsertConversion,
        SecretGenericOpTensorExtractConversion, ConvertTruthTableOp,
        SecretGenericOpInvConversion, SecretGenericOpAndConversion,
        SecretGenericOpNorConversion, SecretGenericOpNandConversion,
        SecretGenericOpOrConversion, SecretGenericOpXNorConversion,
        SecretGenericOpXorConversion, ConvertSecretCastOp,
        ConvertSecretConcealOp, ConvertAnyContextAware<affine::AffineForOp>,
        ConvertAnyContextAware<affine::AffineYieldOp>, ConvertFromElementsOp,
        SecretGenericOpConversion<tensor::EmptyOp>,
        SecretGenericOpTensorInsertSliceConversion>(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyContextAwarePartialConversion(module, target,
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet cleanupPatterns(context);
    tensor::populateFoldCollapseExtractPatterns(cleanupPatterns);
    cleanupPatterns.add<ResolveUnrealizedConversionCast>(context);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
