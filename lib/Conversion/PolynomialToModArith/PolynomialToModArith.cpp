#include "lib/Conversion/PolynomialToModArith/PolynomialToModArith.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "lib/Conversion/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMAttrs.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/Polynomial.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"             // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {
namespace to_mod_arith {

using namespace mlir::polynomial;

#define DEBUG_TYPE "polynomial-to-mod-arith"

#define GEN_PASS_DEF_POLYNOMIALTOMODARITH
#include "lib/Conversion/PolynomialToModArith/PolynomialToModArith.h.inc"

RankedTensorType convertPolynomialType(PolynomialType type) {
  RingAttr attr = type.getRing();
  // We must remove the ring attribute on the tensor, since the
  // unrealized_conversion_casts cannot carry the poly.ring attribute
  // through.
  auto degree = attr.getPolynomialModulus().getPolynomial().getDegree();
  return RankedTensorType::get({degree}, attr.getCoefficientType());
}

class PolynomialToModArithTypeConverter : public TypeConverter {
 public:
  PolynomialToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PolynomialType type) -> Type {
      return convertPolynomialType(type);
    });

    // Convert from a tensor type to a poly type: use from_tensor
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<FromTensorOp>(loc, type, inputs[0]).getResult();
    });

    // Convert from a tensor type to a poly type: use from_tensor
    addArgumentMaterialization([](OpBuilder &builder, Type type,
                                  ValueRange inputs, Location loc) -> Value {
      return builder.create<FromTensorOp>(loc, type, inputs[0]).getResult();
    });

    // Convert from a poly type to a tensor type: use to_tensor
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<ToTensorOp>(loc, type, inputs[0]).getResult();
    });
  }
};

// boilerplate code in one place
template <typename SourceOp>
struct ConvertCommon {
  using OpAdaptor = typename SourceOp::Adaptor;

  ConvertCommon(SourceOp op, OpAdaptor adaptor,
                const TypeConverter *typeConverter) {
    // all current implemented Op use result as the source of polynomial type
    resultType = op.getResult().getType();
    polynomialType = cast<PolynomialType>(resultType);

    ringAttr = polynomialType.getRing();

    coefficientType = ringAttr.getCoefficientType();
    coefficientTypeWidth = coefficientType.getIntOrFloatBitWidth();
    coefficientIntegerType = dyn_cast<IntegerType>(coefficientType);
    coefficientFloatType = dyn_cast<FloatType>(coefficientType);
    // coefficient natural modulus is the power-of-two modulus
    // e.g. 66 for i64 as one for sign bit and one for 2 ** 64
    coefficientNaturalModulus =
        APInt::getOneBitSet(coefficientTypeWidth + 2, coefficientTypeWidth);

    coefficientModulus = ringAttr.getCoefficientModulus();
    polynomialModulus = ringAttr.getPolynomialModulus();

    coefficientModulusIsNotNatural = false;
    if (coefficientModulus) {
      coefficientModulusValue = coefficientModulus.getValue();
      coefficientModulusWidth = coefficientModulusValue.getActiveBits();
      // ringAttr verifier requires coefficientModulus <=
      // coefficientNaturalModulus
      coefficientModulusIsNotNatural =
          coefficientModulusWidth <= coefficientTypeWidth;
    } else if (coefficientIntegerType) {
      // if modulus is not specified, lowering pass can choose
      // to naturally wrap around the underlying type
      coefficientModulusValue = coefficientNaturalModulus;
      coefficientModulusWidth = coefficientModulusValue.getActiveBits();
    }

    // tensor related
    resultTypeConverted = typeConverter->convertType(resultType);
    resultTensorType = cast<RankedTensorType>(resultTypeConverted);
    resultShape = resultTensorType.getShape()[0];
    resultElementType = resultTensorType.getElementType();
    resultElementWidth = resultElementType.getIntOrFloatBitWidth();
    resultElementIntegerType = dyn_cast<IntegerType>(resultElementType);
    resultElementFloatType = dyn_cast<FloatType>(resultElementType);

    llvm::TypeSwitch<Operation &>(*op)
        .Case<FromTensorOp>(
            [&](auto op) { inputType = op.getOperand().getType(); })
        .template Case<AddOp, SubOp, MulScalarOp>(
            [&](auto op) { inputType = op.getOperands()[0].getType(); });
    if (inputType) {
      inputTypeConverted = typeConverter->convertType(inputType);
      inputTensorType = cast<RankedTensorType>(inputTypeConverted);
      inputShape = inputTensorType.getShape()[0];
      inputElementType = inputTensorType.getElementType();
      inputElementWidth = inputElementType.getIntOrFloatBitWidth();
      inputElementIntegerType = dyn_cast<IntegerType>(inputElementType);
      inputElementFloatType = dyn_cast<FloatType>(inputElementType);
    }
  }

  // polynomial type related
  PolynomialType polynomialType;

  RingAttr ringAttr;

  Type coefficientType;
  unsigned coefficientTypeWidth;
  IntegerType coefficientIntegerType;
  FloatType coefficientFloatType;

  IntegerAttr coefficientModulus;
  IntPolynomialAttr polynomialModulus;

  APInt coefficientNaturalModulus;
  APInt coefficientModulusValue;
  bool coefficientModulusIsNotNatural;
  unsigned coefficientModulusWidth;

  // tensor related
  Type resultType;
  Type resultTypeConverted;
  RankedTensorType resultTensorType;
  int64_t resultShape;
  Type resultElementType;
  unsigned resultElementWidth;
  IntegerType resultElementIntegerType;
  FloatType resultElementFloatType;

  Type inputType;
  Type inputTypeConverted;
  RankedTensorType inputTensorType;
  int64_t inputShape;
  Type inputElementType;
  unsigned inputElementWidth;
  IntegerType inputElementIntegerType;
  FloatType inputElementFloatType;
};

template <typename Op>
Value modArithReduceOp(ConvertCommon<Op> &c, ImplicitLocOpBuilder &b,
                       ConversionPatternRewriter &rewriter, Value v,
                       int64_t shape = 0) {
  // why not c.resultTensorType here
  // special case for inputShape != resultShape for FromTensor
  auto resultTensorType = RankedTensorType::get(shape, c.coefficientType);
  Type resultType = shape ? Type(resultTensorType) : Type(c.coefficientType);

  // Extend for mod_arith::reduceOp in case modulus as large as underlying
  // type
  if (c.resultElementWidth == c.coefficientModulusWidth) {
    auto reduceIntegerType =
        rewriter.getIntegerType(c.coefficientModulusWidth + 1);
    auto reduceTensorType = RankedTensorType::get(shape, reduceIntegerType);
    Type reduceType = shape ? Type(reduceTensorType) : Type(reduceIntegerType);

    v = b.create<arith::ExtSIOp>(reduceType, v);
    v = b.create<mod_arith::ReduceOp>(reduceType, v, c.coefficientModulus);
    v = b.create<arith::TruncIOp>(resultType, v);
  } else if (c.coefficientModulusWidth != 0 &&
             c.resultElementWidth > c.coefficientModulusWidth) {
    v = b.create<mod_arith::ReduceOp>(resultType, v, c.coefficientModulus);
  }
  // else float or natural power of two modulus larger than underlying type.
  // no operation needed
  return v;
}

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ConvertCommon<FromTensorOp> c(op, adaptor, typeConverter);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getInput();

    // Extend element type if needed.
    if (c.inputElementWidth != c.resultElementWidth) {
      // FromTensorOp verifies that the coefficient tensor's elements fit into
      // the polynomial.
      assert(c.inputElementWidth < c.resultElementWidth);

      coeffValue = b.create<arith::ExtSIOp>(
          RankedTensorType::get(c.inputShape, c.resultElementIntegerType),
          coeffValue);
    }

    coeffValue = modArithReduceOp<FromTensorOp>(c, b, rewriter, coeffValue,
                                                c.inputShape);

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (c.inputShape < c.resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(c.resultShape - c.inputShape));
      coeffValue = b.create<tensor::PadOp>(
          c.resultTypeConverted, coeffValue, low, high,
          b.create<arith::ConstantOp>(
              rewriter.getIntegerAttr(c.resultElementType, 0)),
          /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  ConvertToTensor(mlir::MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ConvertCommon<ConstantOp> c(op, adaptor, typeConverter);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto intAttr = dyn_cast<TypedIntPolynomialAttr>(op.getValue());
    auto floatAttr = dyn_cast<TypedFloatPolynomialAttr>(op.getValue());
    if (!intAttr && !floatAttr) return failure();

    SmallVector<Attribute> coeffs;
    coeffs.reserve(c.resultShape);
    // This is inefficient for large-degree polys, but as of this writing we
    // don't have a lowering that uses a sparse representation.
    for (size_t i = 0; i < c.resultShape; ++i) {
      if (intAttr) {
        coeffs.push_back(rewriter.getIntegerAttr(c.resultElementType, 0));
      } else {
        coeffs.push_back(rewriter.getFloatAttr(c.resultElementType, 0.0));
      }
    }

    // WARNING: if you don't store the IntPolynomial as an intermediate value
    // before iterating over the terms, you will get a user-after-free bug.
    // See the "Temporary range expression" section in
    // https://en.cppreference.com/w/cpp/language/range-for
    if (intAttr) {
      const IntPolynomial &poly = intAttr.getValue().getPolynomial();
      for (const auto &term : poly.getTerms()) {
        int64_t idx = term.getExponent().getSExtValue();
        APInt coeff = term.getCoefficient().sextOrTrunc(c.resultElementWidth);
        coeffs[idx] = rewriter.getIntegerAttr(c.resultElementType, coeff);
      }
    } else {
      const FloatPolynomial &poly = floatAttr.getValue().getPolynomial();
      for (const auto &term : poly.getTerms()) {
        int64_t idx = term.getExponent().getSExtValue();
        // FIXME: only supports f64 now as TypedFloatPolynomialAttr only
        // parses f64 inputs
        APFloat coeff = term.getCoefficient();
        coeffs[idx] = rewriter.getFloatAttr(c.resultElementType, coeff);
      }
    }

    Value constantOp = b.create<arith::ConstantOp>(
        DenseElementsAttr::get(c.resultTensorType, coeffs));

    constantOp = modArithReduceOp(c, b, rewriter, constantOp, c.resultShape);

    rewriter.replaceOp(op, constantOp);
    return success();
  }
};

template <typename Op, typename ModArithOp, typename ArithIOp,
          typename ArithFOp>
Value modArithBinaryOp(ConvertCommon<Op> &c, ImplicitLocOpBuilder &b, Value lhs,
                       Value rhs) {
  if (c.coefficientModulusIsNotNatural) {
    return b.create<ModArithOp>(lhs, rhs, c.coefficientModulus);
  } else if (c.coefficientIntegerType) {
    return b.create<ArithIOp>(lhs, rhs);
  } else {
    return b.create<ArithFOp>(lhs, rhs);
  }
}

template <typename Op>
const auto modArithAddOp =
    modArithBinaryOp<Op, mod_arith::AddOp, arith::AddIOp, arith::AddFOp>;
template <typename Op>
const auto modArithSubOp =
    modArithBinaryOp<Op, mod_arith::SubOp, arith::SubIOp, arith::SubFOp>;
template <typename Op>
const auto modArithMulOp =
    modArithBinaryOp<Op, mod_arith::MulOp, arith::MulIOp, arith::MulFOp>;

struct ConvertMulScalar : public OpConversionPattern<MulScalarOp> {
  ConvertMulScalar(mlir::MLIRContext *context)
      : OpConversionPattern<MulScalarOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulScalarOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ConvertCommon c(op, adaptor, this->getTypeConverter());

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value scalar = adaptor.getScalar();

    scalar = modArithReduceOp<MulScalarOp>(c, b, rewriter, scalar);

    // MulScalarOp verifier enforces that the input has the same type as the
    // polynomial ring's coefficient type.
    auto tensor = b.create<tensor::SplatOp>(c.resultTensorType, scalar);

    auto mulOp =
        modArithMulOp<MulScalarOp>(c, b, adaptor.getPolynomial(), tensor);
    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

template <typename Op, auto &modArithOpFunc>
struct ConvertAddSub : public OpConversionPattern<Op> {
  ConvertAddSub(mlir::MLIRContext *context)
      : OpConversionPattern<Op>(context) {}

  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ConvertCommon<Op> c(op, adaptor, this->getTypeConverter());

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto addOp = modArithOpFunc(c, b, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, addOp);
    return success();
  }
};

struct PolynomialToModArith
    : impl::PolynomialToModArithBase<PolynomialToModArith> {
  using PolynomialToModArithBase::PolynomialToModArithBase;

  void runOnOperation() override;
};

void PolynomialToModArith::runOnOperation() {
  MLIRContext *context = &getContext();

  ModuleOp module = getOperation();
  ConversionTarget target(*context);
  PolynomialToModArithTypeConverter typeConverter(context);

  target.addIllegalOp<AddOp, SubOp, MulScalarOp, FromTensorOp, ToTensorOp,
                      ConstantOp>();
  RewritePatternSet patterns(context);

  patterns.add<ConvertFromTensor, ConvertToTensor, ConvertConstant,
               ConvertAddSub<AddOp, modArithAddOp<AddOp>>,
               ConvertAddSub<SubOp, modArithSubOp<SubOp>>, ConvertMulScalar>(
      typeConverter, context);
  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace to_mod_arith
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
