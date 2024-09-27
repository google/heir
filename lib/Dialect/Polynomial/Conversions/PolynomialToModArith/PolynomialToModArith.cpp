#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "lib/Utils/ConversionUtils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMAttrs.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/Polynomial.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
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
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h.inc"

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
struct CommonConversionInfo {
  using OpAdaptor = typename SourceOp::Adaptor;

  CommonConversionInfo(SourceOp op, OpAdaptor adaptor,
                       const TypeConverter *typeConverter) {
    // all current implemented Op use result as the source of polynomial type
    Type resultType = op.getResult().getType();

    polynomialType = cast<PolynomialType>(resultType);

    ringAttr = polynomialType.getRing();

    coefficientType = ringAttr.getCoefficientType();
    coefficientTypeWidth = coefficientType.getIntOrFloatBitWidth();
    coefficientIntegerType = dyn_cast<IntegerType>(coefficientType);
    coefficientFloatType = dyn_cast<FloatType>(coefficientType);

    coefficientModulus = ringAttr.getCoefficientModulus();
    polynomialModulus = ringAttr.getPolynomialModulus();

    if (coefficientModulus == nullptr) {
      // coefficient natural modulus is the power-of-two modulus
      // e.g. 66 for i64 as one for sign bit and one for 2 ** 64
      APInt coefficientNaturalModulus =
          APInt::getOneBitSet(coefficientTypeWidth + 2, coefficientTypeWidth);
      MLIRContext *context = resultType.getContext();
      IntegerType coefficientModulusIntType =
          IntegerType::get(context, coefficientTypeWidth + 2);
      coefficientModulus = IntegerAttr::get(coefficientModulusIntType,
                                            coefficientNaturalModulus);
    }
    coefficientModulusValue = coefficientModulus.getValue();
    coefficientModulusWidth = coefficientModulusValue.getActiveBits();

    // tensor related
    tensorType = cast<RankedTensorType>(typeConverter->convertType(resultType));
    tensorShape = tensorType.getShape()[0];
    tensorElementType = tensorType.getElementType();
    tensorElementWidth = tensorElementType.getIntOrFloatBitWidth();
    tensorElementIntegerType = dyn_cast<IntegerType>(tensorElementType);
    tensorElementFloatType = dyn_cast<FloatType>(tensorElementType);
  }

  // polynomial related
  PolynomialType polynomialType;

  RingAttr ringAttr;

  Type coefficientType;
  unsigned coefficientTypeWidth;
  IntegerType coefficientIntegerType;
  FloatType coefficientFloatType;

  IntegerAttr coefficientModulus;
  IntPolynomialAttr polynomialModulus;

  APInt coefficientModulusValue;
  unsigned coefficientModulusWidth;

  // tensor related
  RankedTensorType tensorType;
  int64_t tensorShape;
  Type tensorElementType;
  unsigned tensorElementWidth;
  IntegerType tensorElementIntegerType;
  FloatType tensorElementFloatType;
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    CommonConversionInfo<FromTensorOp> c(op, adaptor, typeConverter);

    RankedTensorType inputTensorType =
        cast<RankedTensorType>(op.getOperand().getType());
    int64_t inputShape = inputTensorType.getShape()[0];
    Type inputElementType = inputTensorType.getElementType();
    unsigned inputElementWidth = inputElementType.getIntOrFloatBitWidth();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getInput();

    // Extend element type if needed.
    if (inputElementWidth != c.tensorElementWidth) {
      // FromTensorOp verifies that the coefficient tensor's elements fit into
      // the polynomial.
      assert(inputElementWidth < c.tensorElementWidth);

      // FromTensorOp defined to take ranked tensor of integer values
      coeffValue = b.create<arith::ExtSIOp>(
          RankedTensorType::get(inputShape, c.tensorElementIntegerType),
          coeffValue);
    }

    coeffValue =
        b.create<mod_arith::ReduceOp>(coeffValue, c.coefficientModulus);

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < c.tensorShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(c.tensorShape - inputShape));
      coeffValue = b.create<tensor::PadOp>(
          c.tensorType, coeffValue, low, high,
          b.create<arith::ConstantOp>(
              rewriter.getIntegerAttr(c.tensorElementType, 0)),
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
    CommonConversionInfo<ConstantOp> c(op, adaptor, typeConverter);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto intAttr = dyn_cast<TypedIntPolynomialAttr>(op.getValue());
    auto floatAttr = dyn_cast<TypedFloatPolynomialAttr>(op.getValue());
    if (!intAttr && !floatAttr) return failure();

    SmallVector<Attribute> coeffs;
    coeffs.reserve(c.tensorShape);
    // This is inefficient for large-degree polys, but as of this writing we
    // don't have a lowering that uses a sparse representation.
    for (size_t i = 0; i < c.tensorShape; ++i) {
      if (intAttr) {
        coeffs.push_back(rewriter.getIntegerAttr(c.tensorElementType, 0));
      } else {
        coeffs.push_back(rewriter.getFloatAttr(c.tensorElementType, 0.0));
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
        APInt coeff = term.getCoefficient().sextOrTrunc(c.tensorElementWidth);
        coeffs[idx] = rewriter.getIntegerAttr(c.tensorElementType, coeff);
      }
    } else {
      const FloatPolynomial &poly = floatAttr.getValue().getPolynomial();
      for (const auto &term : poly.getTerms()) {
        int64_t idx = term.getExponent().getSExtValue();
        // FIXME: only supports f64 now as TypedFloatPolynomialAttr only
        // parses f64 inputs
        APFloat coeff = term.getCoefficient();
        coeffs[idx] = rewriter.getFloatAttr(c.tensorElementType, coeff);
      }
    }

    Value constantOp = b.create<arith::ConstantOp>(
        DenseElementsAttr::get(c.tensorType, coeffs));

    if (intAttr) {
      constantOp =
          b.create<mod_arith::ReduceOp>(constantOp, c.coefficientModulus);
    }

    rewriter.replaceOp(op, constantOp);
    return success();
  }
};

template <typename Op, typename ModArithOp, typename ArithFOp>
Value modArithBinaryOp(CommonConversionInfo<Op> &c, ImplicitLocOpBuilder &b,
                       Value lhs, Value rhs) {
  if (c.coefficientIntegerType) {
    return b.create<ModArithOp>(lhs, rhs, c.coefficientModulus);
  } else {
    return b.create<ArithFOp>(lhs, rhs);
  }
}

template <typename Op>
const auto modArithAddOp =
    modArithBinaryOp<Op, mod_arith::AddOp, arith::AddFOp>;
template <typename Op>
const auto modArithSubOp =
    modArithBinaryOp<Op, mod_arith::SubOp, arith::SubFOp>;
template <typename Op>
const auto modArithMulOp =
    modArithBinaryOp<Op, mod_arith::MulOp, arith::MulFOp>;

struct ConvertMulScalar : public OpConversionPattern<MulScalarOp> {
  ConvertMulScalar(mlir::MLIRContext *context)
      : OpConversionPattern<MulScalarOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulScalarOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    CommonConversionInfo c(op, adaptor, this->getTypeConverter());

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value scalar = b.create<mod_arith::ReduceOp>(adaptor.getScalar(),
                                                 c.coefficientModulus);

    // MulScalarOp verifier enforces that the input has the same type as the
    // polynomial ring's coefficient type.
    auto tensor = b.create<tensor::SplatOp>(c.tensorType, scalar);

    auto mulOp =
        modArithMulOp<MulScalarOp>(c, b, adaptor.getPolynomial(), tensor);
    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    CommonConversionInfo c(op, adaptor, this->getTypeConverter());

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto addOp = modArithAddOp<AddOp>(c, b, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, addOp);
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  ConvertSub(mlir::MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    CommonConversionInfo c(op, adaptor, this->getTypeConverter());

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto subOp = modArithSubOp<SubOp>(c, b, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, subOp);
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

  patterns.add<ConvertFromTensor, ConvertToTensor, ConvertConstant, ConvertAdd,
               ConvertSub, ConvertMulScalar>(typeConverter, context);
  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace to_mod_arith
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
