#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMAttrs.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
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

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace polynomial {

using namespace mlir::heir::polynomial;
using namespace mlir::heir::mod_arith;

#define DEBUG_TYPE "polynomial-to-mod-arith"

#define GEN_PASS_DEF_POLYNOMIALTOMODARITH
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h.inc"

// Callback type for getting pre-generated FuncOp implementing
// helper functions for various lowerings.
using GetFuncCallbackTy = function_ref<func::FuncOp(FunctionType, RingAttr)>;

RankedTensorType convertPolynomialType(PolynomialType type) {
  RingAttr attr = type.getRing();
  // We must remove the ring attribute on the tensor, since the
  // unrealized_conversion_casts cannot carry the poly.ring attribute
  // through.
  auto degree = attr.getPolynomialModulus().getPolynomial().getDegree();
  return RankedTensorType::get({degree}, attr.getCoefficientType());
}

struct CommonConversionInfo {
  PolynomialType polynomialType;
  RingAttr ringAttr;

  Type coefficientType;
  Type coefficientStorageType;

  APInt coefficientModulusValue;
  unsigned coefficientModulusWidth;

  // Poly -> tensor converted type
  RankedTensorType tensorType;
};

FailureOr<CommonConversionInfo> getCommonConversionInfo(
    Operation* op, const TypeConverter* typeConverter,
    std::optional<Type> polyType = std::nullopt) {
  // Most ops have a single result type that is a polynomial
  PolynomialType polyTy;
  if (polyType.has_value()) {
    polyTy = dyn_cast<PolynomialType>(polyType.value());
  } else {
    polyTy = dyn_cast<PolynomialType>(op->getResult(0).getType());
  }

  if (!polyTy) {
    op->emitError(
        "Can't directly lower for a tensor of polynomials. "
        "First run --convert-elementwise-to-affine.");
    return failure();
  }

  CommonConversionInfo info;
  info.polynomialType = polyTy;
  info.ringAttr = info.polynomialType.getRing();
  info.coefficientType = info.ringAttr.getCoefficientType();
  info.tensorType = cast<RankedTensorType>(typeConverter->convertType(polyTy));

  FailureOr<Type> res =
      llvm::TypeSwitch<Type, FailureOr<Type>>(info.coefficientType)
          .Case<IntegerType>([&](auto intTy) { return intTy; })
          .Case<ModArithType>(
              [&](ModArithType intTy) { return intTy.getModulus().getType(); })
          .Default([&](Type ty) { return failure(); });
  if (failed(res)) {
    assert(false && "unsupported coefficient type");
  }
  info.coefficientStorageType = res.value();
  return std::move(info);
}

Value getConstantCoefficient(Type type, int64_t value,
                             ImplicitLocOpBuilder& builder) {
  return llvm::TypeSwitch<Type, Value>(type)
      .Case<IntegerType>([&](auto intTy) {
        return arith::ConstantOp::create(builder,
                                         builder.getIntegerAttr(intTy, value));
      })
      .Case<ModArithType>([&](ModArithType modTy) {
        return mod_arith::ConstantOp::create(
            builder, modTy,
            IntegerAttr::get(modTy.getModulus().getType(), value));
      })
      .Default([&](Type ty) {
        assert(false && "unsupported coefficient type");
        return Value();
      });
}

std::pair<APInt, APInt> extendWidthsToLargest(const APInt& a, const APInt& b) {
  unsigned width = std::max(a.getBitWidth(), b.getBitWidth());
  return {a.zextOrTrunc(width), b.zextOrTrunc(width)};
}

class PolynomialToModArithTypeConverter : public TypeConverter {
 public:
  PolynomialToModArithTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PolynomialType type) -> Type {
      return convertPolynomialType(type);
    });

    // We don't include any custom materialization ops because this lowering is
    // all done in a single pass. The dialect conversion framework works by
    // resolving intermediate (mid-pass) type conflicts by inserting
    // unrealized_conversion_cast ops, and only converting those to custom
    // materializations if they persist at the end of the pass. In our case,
    // we'd only need to use custom materializations if we split this lowering
    // across multiple passes.
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  ConvertFromTensor(mlir::MLIRContext* context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto resultShape = typeInfo.tensorType.getShape()[0];
    auto resultEltTy = typeInfo.tensorType.getElementType();
    auto inputTensorTy = op.getInput().getType();
    auto inputShape = inputTensorTy.getShape()[0];

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getInput();

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(resultShape - inputShape));

      auto padValue = getConstantCoefficient(resultEltTy, 0, b);
      coeffValue = tensor::PadOp::create(b, typeInfo.tensorType, coeffValue,
                                         low, high, padValue,
                                         /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  ConvertToTensor(mlir::MLIRContext* context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  ConvertConstant(mlir::MLIRContext* context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto attr = dyn_cast<TypedIntPolynomialAttr>(op.getValue());
    if (!attr) return failure();
    SmallVector<Attribute> coeffs;
    Type eltStorageType = typeInfo.coefficientStorageType;

    // Create all the attributes as arith types since mod_arith.constant
    // doesn't support tensor attribute inputs. Instead we
    // mod_arith.encapsulate them.
    //
    // This is inefficient for large-degree polys, but as of this writing we
    // don't have a lowering that uses a sparse representation.
    unsigned numTerms = typeInfo.tensorType.getShape()[0];
    coeffs.reserve(numTerms);
    for (size_t i = 0; i < numTerms; ++i) {
      coeffs.push_back(IntegerAttr::get(eltStorageType, 0));
    }

    // WARNING: if you don't store the IntPolynomial as an intermediate value
    // before iterating over the terms, you will get a use-after-free bug. See
    // the "Temporary range expression" section in
    // https://en.cppreference.com/w/cpp/language/range-for
    const IntPolynomial& poly = attr.getValue().getPolynomial();
    for (const auto& term : poly.getTerms()) {
      int64_t idx = term.getExponent().getSExtValue();
      APInt coeff = term.getCoefficient();
      // If the coefficient type is a mod_arith type, then we need to ensure
      // the input is normalized properly to the modulus. I.e., if the
      // polynomial coefficient is a literal -1 and the mod_arith modulus is 7
      // : i32, then -1 equiv 6 mod 7, but -1 as an i32 is 2147483647 equiv 1
      // mod 7.
      if (auto modArithType =
              dyn_cast<ModArithType>(typeInfo.coefficientType)) {
        APInt modulus = modArithType.getModulus().getValue();
        // APInt srem gives remainder with sign matching the sign of the
        // context argument (here, it's the sign of coeff)
        coeff = coeff.sextOrTrunc(modulus.getBitWidth()).srem(modulus);
        if (coeff.isNegative()) {
          // We need to add the modulus to get the positive remainder.
          coeff += modulus;
        }
        assert(coeff.sge(0));
      }
      coeffs[idx] = IntegerAttr::get(eltStorageType, coeff.getSExtValue());
    }

    return llvm::TypeSwitch<Type, LogicalResult>(typeInfo.coefficientType)
        .Case<IntegerType>([&](auto intTy) {
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(
              op, DenseElementsAttr::get(typeInfo.tensorType, coeffs));
          return success();
        })
        .Case<ModArithType>([&](ModArithType intTy) {
          auto intTensorType = RankedTensorType::get(
              typeInfo.tensorType.getShape(), intTy.getModulus().getType());
          auto constOp = arith::ConstantOp::create(
              b, DenseElementsAttr::get(intTensorType, coeffs));
          rewriter.replaceOpWithNewOp<mod_arith::EncapsulateOp>(
              op, typeInfo.tensorType, constOp.getResult());
          return success();
        })
        .Default([&](Type ty) {
          op.emitError("unsupported coefficient type: ") << ty;
          return failure();
        });
    return success();
  }
};

struct ConvertMonomial : public OpConversionPattern<MonomialOp> {
  ConvertMonomial(mlir::MLIRContext* context)
      : OpConversionPattern<MonomialOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MonomialOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto storageTensorType = RankedTensorType::get(
        typeInfo.tensorType.getShape(), typeInfo.coefficientStorageType);
    auto tensor = arith::ConstantOp::create(
        b, DenseElementsAttr::get(
               storageTensorType,
               b.getIntegerAttr(typeInfo.coefficientStorageType, 0)));

    Value result = tensor.getResult();
    if (isa<ModArithType>(
            typeInfo.polynomialType.getRing().getCoefficientType())) {
      result = mod_arith::EncapsulateOp::create(b, typeInfo.tensorType, tensor)
                   .getResult();
    }
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getCoefficient(),
                                                  result, adaptor.getDegree());
    return success();
  }
};

struct ConvertMulScalar : public OpConversionPattern<MulScalarOp> {
  ConvertMulScalar(mlir::MLIRContext* context)
      : OpConversionPattern<MulScalarOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulScalarOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto coeffType = dyn_cast<ModArithType>(typeInfo.coefficientType);
    if (!coeffType) {
      op.emitError("expected coefficient type to be mod_arith type");
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // SplatOp only accepts integer/float inputs, so we can't splat a mod_arith
    // directly.
    auto storageTensorType = RankedTensorType::get(
        typeInfo.tensorType.getShape(), coeffType.getModulus().getType());
    auto tensor = tensor::SplatOp::create(
        b,
        mod_arith::ExtractOp::create(b, storageTensorType.getElementType(),
                                     adaptor.getScalar()),
        storageTensorType);
    auto modArithTensor =
        mod_arith::EncapsulateOp::create(b, typeInfo.tensorType, tensor);
    auto mulOp =
        mod_arith::MulOp::create(b, adaptor.getPolynomial(), modArithTensor);
    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

/// Returns true if and only if the polynomial modulus of the ring for this op
/// is of the form x^n - 1 for some n. This is "cyclic" in the sense that
/// multiplication by a monomial corresponds to a cyclic shift of the
/// coefficients.
bool hasCyclicModulus(MonicMonomialMulOp op) {
  auto ring = cast<PolynomialType>(op.getInput().getType()).getRing();
  IntPolynomial ideal = ring.getPolynomialModulus().getPolynomial();
  auto idealTerms = ideal.getTerms();
  APInt constantCoeff = idealTerms[0].getCoefficient();
  return idealTerms.size() == 2 &&
         (constantCoeff == APInt(constantCoeff.getBitWidth(), -1) &&
          idealTerms[0].getExponent().isZero()) &&
         (idealTerms[1].getCoefficient().isOne());
}

// Implement rotation via tensor.insert_slice
struct ConvertMonicMonomialMul
    : public OpConversionPattern<MonicMonomialMulOp> {
  ConvertMonicMonomialMul(mlir::MLIRContext* context)
      : OpConversionPattern<MonicMonomialMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MonicMonomialMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!hasCyclicModulus(op)) {
      // When upstreaming, this should be expanded to support all inputs.
      InFlightDiagnostic diag =
          op.emitError()
          << "unsupported ideal for monic monomial multiplication:"
          << cast<PolynomialType>(op.getInput().getType())
                 .getRing()
                 .getPolynomialModulus();
      diag.attachNote() << "expected x**n - 1 for some n";
      return diag;
    }

    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // In general, a rotation would correspond to multiplication by x^n,
    // which requires a modular reduction step. But because the verifier
    // requires the ring to have a specific structure (x^n - 1), this op
    // can be implemented as a cyclic shift with wraparound.
    auto outputTensorContainer =
        tensor::EmptyOp::create(b, typeInfo.tensorType.getShape(),
                                typeInfo.tensorType.getElementType());

    RankedTensorType dynamicSliceType = RankedTensorType::get(
        ShapedType::kDynamic, typeInfo.tensorType.getElementType());

    // split the tensor into two pieces at index N - rotation_amount
    // e.g., if rotation_amount is 2,
    //
    //   [0, 1, 2, 3, 4 | 5, 6]
    //
    // the resulting output is
    //
    //   [5, 6 | 0, 1, 2, 3, 4]
    auto constTensorDim = arith::ConstantOp::create(
        b, b.getIndexType(), b.getIndexAttr(typeInfo.tensorType.getShape()[0]));
    auto splitPoint =
        arith::SubIOp::create(b, constTensorDim, adaptor.getMonomialDegree());

    SmallVector<OpFoldResult> firstHalfExtractOffsets{b.getIndexAttr(0)};
    SmallVector<OpFoldResult> firstHalfExtractSizes{splitPoint.getResult()};
    SmallVector<OpFoldResult> strides{b.getIndexAttr(1)};
    auto firstHalfExtractOp = tensor::ExtractSliceOp::create(
        b,
        /*resultType=*/dynamicSliceType,
        /*source=*/adaptor.getInput(), firstHalfExtractOffsets,
        firstHalfExtractSizes, strides);

    SmallVector<OpFoldResult> secondHalfExtractOffsets{splitPoint.getResult()};
    SmallVector<OpFoldResult> secondHalfExtractSizes{
        adaptor.getMonomialDegree()};
    auto secondHalfExtractOp = tensor::ExtractSliceOp::create(
        b,
        /*resultType=*/dynamicSliceType,
        /*source=*/adaptor.getInput(), secondHalfExtractOffsets,
        secondHalfExtractSizes, strides);

    SmallVector<OpFoldResult> firstHalfInsertOffsets{
        adaptor.getMonomialDegree()};
    SmallVector<OpFoldResult> firstHalfInsertSizes{splitPoint.getResult()};
    auto firstHalfInsertOp = tensor::InsertSliceOp::create(
        b,
        /*source=*/firstHalfExtractOp.getResult(),
        /*dest=*/outputTensorContainer.getResult(), firstHalfInsertOffsets,
        firstHalfInsertSizes, strides);

    SmallVector<OpFoldResult> secondHalfInsertOffsets{b.getIndexAttr(0)};
    SmallVector<OpFoldResult> secondHalfInsertSizes{
        adaptor.getMonomialDegree()};
    auto secondHalfInsertOp = tensor::InsertSliceOp::create(
        b,
        /*source=*/secondHalfExtractOp.getResult(),
        /*dest=*/firstHalfInsertOp.getResult(), secondHalfInsertOffsets,
        secondHalfInsertSizes, strides);

    rewriter.replaceOp(op, secondHalfInsertOp.getResult());
    return success();
  }
};

struct ConvertLeadingTerm : public OpConversionPattern<LeadingTermOp> {
  ConvertLeadingTerm(mlir::MLIRContext* context)
      : OpConversionPattern<LeadingTermOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LeadingTermOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffs = adaptor.getInput();
    auto tensorType = cast<RankedTensorType>(coeffs.getType());

    auto res =
        getCommonConversionInfo(op, typeConverter, op.getInput().getType());
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto c0 = arith::ConstantOp::create(
        b, b.getIntegerAttr(typeInfo.coefficientStorageType, 0));
    auto c1 = arith::ConstantOp::create(b, b.getIndexAttr(1));
    auto initIndex = arith::ConstantOp::create(
        b, b.getIndexAttr(tensorType.getShape()[0] - 1));

    auto degreeOp = scf::WhileOp::create(
        b,
        /*resultTypes=*/
        TypeRange{b.getIndexType()},
        /*operands=*/ValueRange{initIndex.getResult()},
        /*beforeBuilder=*/
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
          Value index = args[0];
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          auto coeff = tensor::ExtractOp::create(b, coeffs, ValueRange{index});
          auto normalizedCoeff = mod_arith::ReduceOp::create(b, coeff);
          auto extractedCoeff = mod_arith::ExtractOp::create(
              b, typeInfo.coefficientStorageType, normalizedCoeff);
          auto cmpOp = arith::CmpIOp::create(b, arith::CmpIPredicate::eq,
                                             extractedCoeff, c0);
          scf::ConditionOp::create(b, cmpOp.getResult(), index);
        },
        /*afterBuilder=*/
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          Value currentIndex = args[0];
          auto nextIndex =
              arith::SubIOp::create(b, currentIndex, c1.getResult());
          scf::YieldOp::create(b, nextIndex.getResult());
        });
    auto degree = degreeOp.getResult(0);
    auto leadingCoefficient =
        tensor::ExtractOp::create(b, coeffs, ValueRange{degree});
    rewriter.replaceOp(op, ValueRange{degree, leadingCoefficient.getResult()});
    return success();
  }
};

template <typename SourceOp, typename TargetArithOp, typename TargetModArithOp>
struct ConvertPolyBinop : public OpConversionPattern<SourceOp> {
  ConvertPolyBinop(mlir::MLIRContext* context)
      : OpConversionPattern<SourceOp>(context) {}

  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto res = getCommonConversionInfo(op, this->typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    return llvm::TypeSwitch<Type, LogicalResult>(typeInfo.coefficientType)
        .template Case<IntegerType>([&](auto intTy) {
          auto result =
              TargetArithOp::create(b, adaptor.getLhs(), adaptor.getRhs());
          rewriter.replaceOp(op, result);
          return success();
        })
        .template Case<ModArithType>([&](ModArithType intTy) {
          auto result =
              TargetModArithOp::create(b, adaptor.getLhs(), adaptor.getRhs());
          rewriter.replaceOp(op, result);
          return success();
        })
        .Default([&](Type ty) {
          op.emitError("unsupported coefficient type: ") << ty;
          return failure();
        });
  }
};

RankedTensorType polymulOutputTensorType(PolynomialType type) {
  auto convDegree =
      2 * type.getRing().getPolynomialModulus().getPolynomial().getDegree() - 1;
  return RankedTensorType::get({convDegree},
                               type.getRing().getCoefficientType());
}

// Lower polynomial multiplication to a 1D convolution, followed by with a
// modulus reduction in the ring.
struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(const TypeConverter& typeConverter, mlir::MLIRContext* context,
             GetFuncCallbackTy cb)
      : OpConversionPattern<MulOp>(typeConverter, context),
        getFuncOpCallback(cb) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto res = getCommonConversionInfo(op, this->typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();
    auto coeffType = dyn_cast<ModArithType>(typeInfo.coefficientType);
    if (!coeffType) {
      op.emitError("expected coefficient type to be mod_arith type");
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // Implementing a naive polymul operation which is a loop
    //
    // for i = 0, ..., N-1
    //   for j = 0, ..., N-1
    //     c[i+j] += a[i] * b[j]
    //
    RankedTensorType polymulTensorType =
        polymulOutputTensorType(typeInfo.polynomialType);

    SmallVector<utils::IteratorType> iteratorTypes(
        2, utils::IteratorType::parallel);
    AffineExpr d0, d1;
    bindDims(getContext(), d0, d1);
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(2, 0, {d0}),      // i
        AffineMap::get(2, 0, {d1}),      // j
        AffineMap::get(2, 0, {d0 + d1})  // i+j
    };

    auto intStorageType = coeffType.getModulus().getType();
    auto storageTensorType =
        RankedTensorType::get(polymulTensorType.getShape(), intStorageType);
    auto tensor = arith::ConstantOp::create(
        b, DenseElementsAttr::get(storageTensorType,
                                  b.getIntegerAttr(intStorageType, 0)));
    // The tensor of zeros in which to store the naive polymul output from the
    // linalg.generic op below.
    auto polymulOutput =
        mod_arith::EncapsulateOp::create(b, polymulTensorType, tensor);

    auto polyMul = linalg::GenericOp::create(
        b,
        /*resultTypes=*/polymulTensorType,
        /*inputs=*/adaptor.getOperands(),
        /*outputs=*/polymulOutput.getResult(),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          auto lhs = args[0];
          auto rhs = args[1];
          auto accum = args[2];
          auto mulOp = mod_arith::MulOp::create(b, lhs, rhs);
          auto addOp = mod_arith::AddOp::create(b, mulOp, accum);
          linalg::YieldOp::create(b, addOp.getResult());
        });

    auto postReductionType = convertPolynomialType(typeInfo.polynomialType);
    FunctionType funcType = FunctionType::get(
        op.getContext(), {polymulTensorType}, {postReductionType});

    // 2N - 1 sized result tensor -> reduce modulo ideal to get a N sized tensor
    func::FuncOp divMod = getFuncOpCallback(funcType, typeInfo.ringAttr);
    if (!divMod) {
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic& diag) {
        diag << "Missing software implementation for polynomial mod op of type"
             << funcType << " and for ring " << typeInfo.ringAttr;
      });
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, divMod, polyMul.getResult(0));
    return success();
  }

 private:
  GetFuncCallbackTy getFuncOpCallback;
};

struct PolynomialToModArith
    : impl::PolynomialToModArithBase<PolynomialToModArith> {
  using PolynomialToModArithBase::PolynomialToModArithBase;

  void runOnOperation() override;

 private:
  // Generate implementations for operations
  void generateOpImplementations();

  func::FuncOp buildPolynomialModFunc(FunctionType funcType, RingAttr ringAttr);

  // A map containing modular reduction function implementations, generated once
  // at the beginning of this pass based on the ops to be converted, intended to
  // be retrieved by ConvertMul to construct CallOps so that later optimization
  // passes can determine when to inline the implementation.
  DenseMap<std::pair<Type, RingAttr>, func::FuncOp> modImpls;
};

void PolynomialToModArith::generateOpImplementations() {
  ModuleOp module = getOperation();
  module.walk([&](MulOp op) {
    auto polyTy = dyn_cast<PolynomialType>(op.getResult().getType());
    if (!polyTy) {
      op.emitError()
          << "Encountered elementwise polynomial.mul op. The caller must use "
             "convert-elementwise-to-affine pass before lowering polynomial.";
      return WalkResult::interrupt();
    }
    auto convType = polymulOutputTensorType(polyTy);
    auto postReductionType = convertPolynomialType(polyTy);
    FunctionType funcType =
        FunctionType::get(op.getContext(), {convType}, {postReductionType});

    // Generate the software implementation of modular reduction if it has not
    // been generated yet.
    auto key = std::pair(funcType, polyTy.getRing());
    if (!modImpls.count(key)) {
      func::FuncOp modOp = buildPolynomialModFunc(funcType, polyTy.getRing());
      modImpls.insert(std::pair(key, modOp));
    }
    return WalkResult::advance();
  });
}

// Create a software implementation that reduces a polynomial
// modulo a statically known divisor polynomial.
func::FuncOp PolynomialToModArith::buildPolynomialModFunc(FunctionType funcType,
                                                          RingAttr ring) {
  ModuleOp module = getOperation();
  Location loc = module->getLoc();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  // These tensor types are used in the implementation
  //
  //  - The input tensor<2047x!coeff_ty>, e.g., the output of a naive polymul
  //    of two tensor<1024x!coeff_ty>
  //  - The result tensor<1024x!coeff_ty>, e.g., the result after modular
  //    reduction is complete and represents the remainder being accumulated
  RankedTensorType inputType =
      llvm::cast<RankedTensorType>(funcType.getInput(0));
  RankedTensorType resultType =
      llvm::cast<RankedTensorType>(funcType.getResult(0));

  // TODO(#202): this function name probably also needs the input tensor type in
  // the name, or it could conflict with other implementations that have the
  // same cmod+ideal.
  auto coeffTy = ring.getCoefficientType();
  std::string coeffTyId;
  // TODO(#1199): support RNS lowering
  if (auto intTy = dyn_cast<IntegerType>(coeffTy)) {
    coeffTyId = llvm::formatv("i{0}", intTy.getWidth());
  } else if (auto modTy = dyn_cast<ModArithType>(coeffTy)) {
    IntegerType intTy = cast<IntegerType>(modTy.getModulus().getType());
    SmallString<10> modulusStr;
    modTy.getModulus().getValue().toStringUnsigned(modulusStr);
    coeffTyId =
        llvm::formatv("{0}_i{1}", modulusStr, intTy.getIntOrFloatBitWidth());
  }
  std::string funcName =
      llvm::formatv("__heir_poly_mod_{0}_{1}", coeffTyId,
                    ring.getPolynomialModulus().getPolynomial().toIdentifier());

  auto funcOp = func::FuncOp::create(builder, funcName, funcType);
  LLVM::linkage::Linkage inlineLinkage = LLVM::linkage::Linkage::LinkonceODR;
  Attribute linkage =
      LLVM::LinkageAttr::get(builder.getContext(), inlineLinkage);
  funcOp->setAttr("llvm.linkage", linkage);
  funcOp.setPrivate();

  Block* funcBody = funcOp.addEntryBlock();
  Value coeffsArg = funcOp.getArgument(0);

  builder.setInsertionPointToStart(funcBody);

  // Implementing the textbook division algorithm
  //
  // def divmod(poly, divisor):
  //   divisorLC = divisor.leadingCoefficient()
  //   divisorDeg = divisor.degree()
  //   remainder = poly
  //   quotient = Zero()
  //
  //   while remainder.degree() >= divisorDeg:
  //      monomialExponent = remainder.degree() - divisorDeg
  //      monomialDivisor = monomial(
  //        monomialExponent,
  //        remainder.leadingCoefficient() / divisorLC
  //      )
  //      quotient += monomialDivisor
  //      remainder -= monomialDivisor * divisor
  //
  //   return quotient, remainder
  //

  // Implementing the algorithm using poly ops, but the function signature
  // input is in terms of the lowered tensor types, so we need a from_tensor.
  // We also need to pick an appropriate ring, which in our case will be the
  // ring of polynomials mod (x^n - 1).
  std::vector<IntMonomial> monomials;
  // If the input has size N as a tensor, then as a polynomial its max degree is
  // N-1, and we want the ring to be mod (x^N - 1).
  unsigned remRingDegree = inputType.getShape()[0];
  monomials.emplace_back(1, remRingDegree);
  monomials.emplace_back(-1, 0);
  IntPolynomial xnMinusOne = IntPolynomial::fromMonomials(monomials).value();
  IntPolynomialAttr xnMinusOneAttr =
      IntPolynomialAttr::get(&getContext(), xnMinusOne);
  auto remRing = RingAttr::get(coeffTy, xnMinusOneAttr);
  auto remRingPolynomialType = PolynomialType::get(&getContext(), remRing);

  // Start by converting the input tensor back to a poly.
  auto fromTensorOp =
      FromTensorOp::create(builder, remRingPolynomialType, coeffsArg);

  // If the leading coefficient of the divisor has no inverse, we can't do
  // division. The lowering must fail:
  auto divisor = ring.getPolynomialModulus().getPolynomial();
  // TODO(#1199): support RNS lowering
  APInt rawCoeffMod =
      cast<ModArithType>(ring.getCoefficientType()).getModulus().getValue();
  auto [leadingCoef, coeffMod] = extendWidthsToLargest(
      divisor.getTerms().back().getCoefficient(), rawCoeffMod);
  auto leadingCoefInverse = multiplicativeInverse(leadingCoef, coeffMod);
  // returning zero means no inverse was found
  if (leadingCoefInverse.isZero()) {
    signalPassFailure();
  }
  auto divisorLcInverse = getConstantCoefficient(
      coeffTy, leadingCoefInverse.getSExtValue(), builder);
  auto divisorDeg =
      arith::ConstantOp::create(builder, builder.getIndexType(),
                                builder.getIndexAttr(divisor.getDegree()));

  // while remainder.degree() >= divisorDeg:
  auto remainder = fromTensorOp.getResult();
  auto whileOp = scf::WhileOp::create(
      builder,
      /*resultTypes=*/
      remainder.getType(),
      /*operands=*/remainder,
      /*beforeBuilder=*/
      [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
        Value remainder = args[0];
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        // remainder.degree() >= divisorDeg
        auto remainderLt = LeadingTermOp::create(
            b, b.getIndexType(), inputType.getElementType(), remainder);
        auto cmpOp = arith::CmpIOp::create(b, arith::CmpIPredicate::sge,
                                           remainderLt.getDegree(), divisorDeg);
        scf::ConditionOp::create(b, cmpOp, remainder);
      },
      /*afterBuilder=*/
      [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value remainder = args[0];
        // TODO(#97): move this out of the loop when it has ConstantLike trait
        auto divisorOp = ConstantOp::create(
            builder, remRingPolynomialType,
            TypedIntPolynomialAttr::get(remRingPolynomialType, divisor));

        // monomialExponent = remainder.degree() - divisorDeg
        auto ltOp = LeadingTermOp::create(
            b, b.getIndexType(), inputType.getElementType(), remainder);
        auto monomialExponentOp =
            arith::SubIOp::create(b, ltOp.getDegree(), divisorDeg);

        // monomialDivisor = monomial(
        //   monomialExponent, remainder.leadingCoefficient() / divisorLC)
        auto monomialLc = mod_arith::MulOp::create(b, ltOp.getCoefficient(),
                                                   divisorLcInverse);

        // remainder -= monomialDivisor * divisor
        auto scaledDivisor = MulScalarOp::create(b, divisorOp, monomialLc);
        auto remainderIncrement = MonicMonomialMulOp::create(
            b, scaledDivisor, monomialExponentOp.getResult());
        auto nextRemainder = SubOp::create(b, remainder, remainderIncrement);

        scf::YieldOp::create(b, nextRemainder.getResult());
      });

  // The result remainder is still in the larger ring, so we need to convert to
  // the smaller ring.
  auto toTensorOp =
      ToTensorOp::create(builder, inputType, whileOp.getResult(0));

  // Smaller ring has a coefficient modulus that needs to be accounted for.
  // Probably a better way to define a splatted dense elements attr, but either
  // way this should be folded/canonicalized into a single op.
  SmallVector<OpFoldResult> offsets{builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes{
      builder.getIndexAttr(resultType.getShape()[0])};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
  auto extractedTensor = tensor::ExtractSliceOp::create(
      builder, resultType, toTensorOp.getResult(), offsets, sizes, strides);

  func::ReturnOp::create(builder, extractedTensor.getResult());
  return funcOp;
}

// Multiply two integers x, y modulo cmod.
static APInt mulMod(const APInt& _x, const APInt& _y, const APInt& _cmod) {
  assert(_x.getBitWidth() == _y.getBitWidth() &&
         "expected same bitwidth of operands");
  auto intermediateBitwidth = _cmod.getBitWidth() * 2;
  APInt x = _x.zext(intermediateBitwidth);
  APInt y = _y.zext(intermediateBitwidth);
  APInt cmod = _cmod.zext(intermediateBitwidth);
  APInt res = (x * y).urem(cmod);
  return res.trunc(_x.getBitWidth());
}

// Compute the first degree powers of root modulo cmod.
static SmallVector<APInt> precomputeRoots(APInt root, const APInt& cmod,
                                          unsigned degree) {
  APInt baseRoot = root;
  root = 1;
  SmallVector<APInt> vals(degree);
  for (unsigned i = 0; i < degree; i++) {
    vals[i] = root;
    root = mulMod(root, baseRoot, cmod);
  }
  return vals;
}

static Value computeReverseBitOrder(ImplicitLocOpBuilder& b,
                                    RankedTensorType tensorType, Type modType,
                                    Value tensor) {
  unsigned degree = tensorType.getShape()[0];
  double degreeLog = std::log2((double)degree);
  assert(std::floor(degreeLog) == degreeLog &&
         "expected the degree to be a power of 2");

  unsigned indexBitWidth = (unsigned)degreeLog;
  auto indicesType = RankedTensorType::get(tensorType.getShape(),
                                           IndexType::get(b.getContext()));

  SmallVector<APInt> _indices(degree);
  for (unsigned index = 0; index < degree; index++) {
    _indices[index] = APInt(indexBitWidth, index).reverseBits();
  }
  auto indices = arith::ConstantOp::create(
      b, indicesType, DenseElementsAttr::get(indicesType, _indices));

  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  AffineExpr d0;
  bindDims(b.getContext(), d0);
  SmallVector<AffineMap> indexingMaps = {AffineMap::get(1, 0, {d0}),
                                         AffineMap::get(1, 0, {d0})};
  auto out = arith::ConstantOp::create(b, tensorType,
                                       DenseElementsAttr::get(tensorType, 0));
  auto modOut = mod_arith::EncapsulateOp::create(b, modType, out);
  auto shuffleOp = linalg::GenericOp::create(
      b,
      /*resultTypes=*/TypeRange{modType},
      /*inputs=*/ValueRange{indices.getResult()},
      /*outputs=*/ValueRange{modOut.getResult()},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        auto idx = args[0];
        auto elem = tensor::ExtractOp::create(b, tensor, ValueRange{idx});
        linalg::YieldOp::create(b, elem.getResult());
      });
  return shuffleOp.getResult(0);
}

static std::pair<Value, Value> bflyCT(ImplicitLocOpBuilder& b, Value A, Value B,
                                      Value root) {
  auto rootB = mod_arith::MulOp::create(b, B, root);
  auto ctPlus = mod_arith::AddOp::create(b, A, rootB);
  auto ctMinus = mod_arith::SubOp::create(b, A, rootB);
  return {ctPlus, ctMinus};
}

static std::pair<Value, Value> bflyGS(ImplicitLocOpBuilder& b, Value A, Value B,
                                      Value root) {
  auto gsPlus = mod_arith::AddOp::create(b, A, B);
  auto gsMinus = mod_arith::SubOp::create(b, A, B);
  auto gsMinusRoot = mod_arith::MulOp::create(b, gsMinus, root);
  return {gsPlus, gsMinusRoot};
}

template <bool inverse>
static Value fastNTT(ImplicitLocOpBuilder& b, RingAttr ring,
                     PrimitiveRootAttr rootAttr, RankedTensorType tensorType,
                     Type modType, Value input) {
  // Compute the number of stages required to compute the NTT
  auto degree = tensorType.getShape()[0];
  unsigned stages = (unsigned)std::log2((double)degree);

  // Precompute the roots
  auto modArithType = cast<ModArithType>(ring.getCoefficientType());
  APInt cmod = modArithType.getModulus().getValue();
  APInt root = rootAttr.getValue().getValue();
  root = !inverse ? root
                  : multiplicativeInverse(root.zext(cmod.getBitWidth()), cmod)
                        .trunc(root.getBitWidth());
  // Initialize the mod_arith roots constant
  auto rootsType = tensorType.clone({degree});
  Value roots = arith::ConstantOp::create(
      b, rootsType,
      DenseElementsAttr::get(rootsType, precomputeRoots(root, cmod, degree)));
  roots = mod_arith::EncapsulateOp::create(b, modType, roots);

  // Here is a slightly modified implementation of the standard iterative NTT
  // computation using Cooley-Turkey/Gentleman-Sande butterfly. For reader
  // reference: https://doi.org/10.1007/978-3-031-46077-7_22, and,
  // https://doi.org/10.1109/ACCESS.2023.3294446
  //
  // We modify the standard implementation by pre-computing the root
  // exponential values during compilation instead of doing so at runtime.
  //
  // Let roots be a tensor of <n x ix> where roots[i] = \psi^i, n be the
  // degree of the polynomial and inverse denote the direction. Then we
  // implement the following:
  //
  // def fastNTT(coeffs, n, cmod, roots, inverse):
  //  m = inverse ? n : 2             # m denotes the batchSize or stride
  //  r = inverse ? 1 : degree / 2    # r denotes the exponent of the root
  //  for (s = 0; s < log2(n); s++):
  //    for (k = 0; k < n / m; k++):
  //      for (j = 0; j < m / 2; j++):
  //        A = coeffs[k * m + j]
  //        B = coeffs[k * m + j + m / 2]
  //        root = roots[(2 * j + 1) * rootExp]
  //        coeffs[k * m + j], coeffs[k * m + j + m / 2]
  //          = bflyOp(A, B, root, cmod)
  //      end
  //    end
  //    m = inverse ? m / 2 : m * 2
  //    r = inverse ? r * 2 : m / 2
  //  end
  //
  //  where bflyOp is one of:
  //    bflyCT(A, B, root, cmod):
  //      (A + root * B % cmod, A - root * B % cmod)
  //
  //    bflyGS(A, B, root, cmod):
  //      (A + B % cmod, (A - B) * root % cmod)

  // Initialize the variables
  Value initialValue = mod_arith::ReduceOp::create(b, input);
  Value initialBatchSize =
      arith::ConstantIndexOp::create(b, inverse ? degree : 2);
  Value initialRootExp =
      arith::ConstantIndexOp::create(b, inverse ? 1 : degree / 2);
  Value zero = arith::ConstantIndexOp::create(b, 0);
  Value one = arith::ConstantIndexOp::create(b, 1);
  Value two = arith::ConstantIndexOp::create(b, 2);
  Value n = arith::ConstantIndexOp::create(b, degree);

  // Define index affine mappings
  AffineExpr x, y;
  bindDims(b.getContext(), x, y);

  Value stagesLb = zero;
  Value stagesUb = arith::ConstantIndexOp::create(b, stages);
  Value stagesStep = one;

  auto stagesLoop = scf::ForOp::create(
      b, stagesLb, stagesUb, stagesStep,
      /*iterArgs=*/ValueRange{initialValue, initialBatchSize, initialRootExp},
      /*bodyBuilder=*/
      [&](OpBuilder& nestedBuilder, Location nestedLoc, Value index,
          ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value batchSize = args[1];
        Value rootExp = args[2];

        Value innerLb = zero;
        Value innerUb = arith::FloorDivSIOp::create(b, n, batchSize);
        Value innerStep = one;

        auto innerLoop = scf::ForOp::create(
            b, innerLb, innerUb, innerStep,
            /*iterArgs=*/args[0],
            /*bodyBuilder=*/
            [&](OpBuilder& nestedBuilder, Location nestedLoc, Value index,
                ValueRange args) {
              ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
              Value indexK = arith::MulIOp::create(b, batchSize, index);
              Value arithLb = zero;
              Value arithUb = arith::FloorDivSIOp::create(b, batchSize, two);
              Value arithStep = one;

              auto arithLoop = scf::ForOp::create(
                  b, arithLb, arithUb, arithStep, /*iterArgs=*/args[0],
                  /*bodyBuilder=*/
                  [&](OpBuilder& nestedBuilder, Location nestedLoc,
                      Value indexJ, ValueRange args) {
                    ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);

                    Value target = args[0];

                    // Get A: indexJ + indexK
                    Value indexA = arith::AddIOp::create(b, indexJ, indexK);
                    Value A = tensor::ExtractOp::create(b, target, indexA);

                    // Get B: indexA + batchSize // 2
                    Value indexB = arith::AddIOp::create(b, indexA, arithUb);
                    Value B = tensor::ExtractOp::create(b, target, indexB);

                    // Get root: (2 * indexJ + 1) * rootExp
                    Value rootIndex = arith::MulIOp::create(
                        b,
                        arith::AddIOp::create(
                            b, arith::MulIOp::create(b, two, indexJ), one),
                        rootExp);
                    Value root = tensor::ExtractOp::create(b, roots, rootIndex);

                    auto bflyResult =
                        inverse ? bflyGS(b, A, B, root) : bflyCT(b, A, B, root);

                    // Store updated values into accumulator
                    auto insertPlus = tensor::InsertOp::create(
                        b, bflyResult.first, target, indexA);
                    auto insertMinus = tensor::InsertOp::create(
                        b, bflyResult.second, insertPlus, indexB);

                    scf::YieldOp::create(b, insertMinus.getResult());
                  });

              scf::YieldOp::create(b, arithLoop.getResult(0));
            });

        batchSize = inverse
                        ? arith::DivUIOp::create(b, batchSize, two).getResult()
                        : arith::MulIOp::create(b, batchSize, two).getResult();

        rootExp = inverse ? arith::MulIOp::create(b, rootExp, two).getResult()
                          : arith::DivUIOp::create(b, rootExp, two).getResult();

        scf::YieldOp::create(
            b, ValueRange{innerLoop.getResult(0), batchSize, rootExp});
      });

  Value result = stagesLoop.getResult(0);
  if (inverse) {
    APInt degreeInv =
        multiplicativeInverse(APInt(cmod.getBitWidth(), degree), cmod)
            .trunc(root.getBitWidth());
    Value nInv = arith::ConstantOp::create(
        b, rootsType, DenseElementsAttr::get(rootsType, degreeInv));
    nInv = mod_arith::EncapsulateOp::create(b, modType, nInv);
    result = mod_arith::MulOp::create(b, result, nInv);
  }

  return result;
}

struct ConvertNTT : public OpConversionPattern<NTTOp> {
  ConvertNTT(mlir::MLIRContext* context)
      : OpConversionPattern<NTTOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      NTTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto polyTy = dyn_cast<PolynomialType>(op.getInput().getType());
    if (!polyTy) {
      op.emitError(
          "Can't directly lower for a tensor of polynomials. "
          "First run --convert-elementwise-to-affine.");
      return failure();
    }

    if (!op.getRoot()) {
      op.emitError("missing root attribute");
      return failure();
    }

    RingAttr ring = polyTy.getRing();
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto coeffType =
        dyn_cast<ModArithType>(polyTy.getRing().getCoefficientType());
    if (!coeffType) {
      op.emitError("expected coefficient type to be mod_arith type");
      return failure();
    }
    auto coeffStorageType = coeffType.getModulus().getType();
    auto intTensorType =
        RankedTensorType::get(inputType.getShape(), coeffStorageType);
    auto modType = adaptor.getInput().getType();

    // Compute the ntt and extract the values
    Value nttResult = fastNTT<false>(
        b, ring, op.getRoot().value(), intTensorType, modType,
        computeReverseBitOrder(b, intTensorType, modType, adaptor.getInput()));

    // Insert the ring encoding here to the input type
    auto outputType =
        RankedTensorType::get(inputType.getShape(), coeffType, ring);
    auto intResult = tensor::CastOp::create(b, outputType, nttResult);
    rewriter.replaceOp(op, intResult);

    return success();
  }
};

struct ConvertINTT : public OpConversionPattern<INTTOp> {
  ConvertINTT(mlir::MLIRContext* context)
      : OpConversionPattern<INTTOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      INTTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    if (!op.getRoot()) {
      op.emitError("missing root attribute");
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto coeffType = dyn_cast<ModArithType>(typeInfo.coefficientType);
    if (!coeffType) {
      op.emitError("expected coefficient type to be mod_arith type");
      return failure();
    }
    auto coeffStorageType = coeffType.getModulus().getType();
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto intTensorType =
        RankedTensorType::get(inputType.getShape(), coeffStorageType);
    auto modType = typeConverter->convertType(op.getOutput().getType());

    // Remove the encoded ring from input tensor type and convert to mod_arith
    // type
    auto input = tensor::CastOp::create(b, modType, adaptor.getInput());
    auto nttResult = fastNTT<true>(b, typeInfo.ringAttr, op.getRoot().value(),
                                   intTensorType, modType, input);

    auto reversedBitOrder =
        computeReverseBitOrder(b, intTensorType, modType, nttResult);
    rewriter.replaceOp(op, reversedBitOrder);

    return success();
  }
};

struct ConvertKeySwitchInner
    : public OpConversionPattern<polynomial::KeySwitchInnerOp> {
  ConvertKeySwitchInner(mlir::MLIRContext* context)
      : OpConversionPattern<polynomial::KeySwitchInnerOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      polynomial::KeySwitchInnerOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto b = ImplicitLocOpBuilder(op.getLoc(), rewriter);

    RankedTensorType coefficientTensorType =
        cast<RankedTensorType>(adaptor.getValue().getType());

    if (coefficientTensorType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op,
          "Can only lower key_switch_inner when input is a single polynomial");
    }

    // TODO(#2157): enable this for more than just CKKS
    ckks::SchemeParamAttr schemeParamAttr =
        op->getParentOfType<ModuleOp>()->getAttrOfType<ckks::SchemeParamAttr>(
            ckks::CKKSDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      return rewriter.notifyMatchFailure(
          op, "Cannot find scheme param attribute on parent module");
    }

    auto schemeParam =
        ckks::SchemeParam::getSchemeParamFromAttr(schemeParamAttr);

    int64_t partSize = schemeParam.getPi().size();

    if (partSize <= 0) {
      return rewriter.notifyMatchFailure(
          op, "Cannot lower key_switch_inner with empty modulus chain");
    }

    int64_t numFullPartitions = coefficientTensorType.getShape()[0] / partSize;
    int64_t sliceSize = partSize * numFullPartitions;
    int64_t extraPartSize = coefficientTensorType.getShape()[0] - sliceSize;

    // Step 1: partition the coefficients of the polynomial into pieces. Note
    // the partition may not be even, so there may be one leftover part of an
    // uneven size that must be tracked manually through the rest of the
    // lowering.

    // First extract the parts that are divisible and reshape it i.e., a
    // tensor<17xi32> with partitionSize 5 would first slice-extract to
    // tensor<15xi32> then reshape to tensor<3x5xi32>.
    SmallVector<OpFoldResult> offsets(1, b.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes(1, b.getIndexAttr(sliceSize));
    SmallVector<OpFoldResult> strides(1, b.getIndexAttr(1));
    Value extracted = tensor::ExtractSliceOp::create(b, adaptor.getValue(),
                                                     offsets, sizes, strides);

    SmallVector<int64_t> partitionShape = {numFullPartitions, partSize};
    RankedTensorType partitionType = RankedTensorType::get(
        partitionShape, coefficientTensorType.getElementType());
    // Create a dense constant for targetShape
    auto shapeOp = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get(partitionType.getRank(), rewriter.getIndexType()),
        rewriter.getIndexTensorAttr(partitionType.getShape()));

    [[maybe_unused]] Value reshaped =
        tensor::ReshapeOp::create(b, partitionType, extracted, shapeOp);

    // Then extract the trailing piece that doesn't have the same tensor size.
    SmallVector<OpFoldResult> extraPartOffsets(1, b.getIndexAttr(sliceSize));
    SmallVector<OpFoldResult> extraPartSizes(1, b.getIndexAttr(extraPartSize));
    SmallVector<OpFoldResult> extraPartStrides(1, b.getIndexAttr(1));
    [[maybe_unused]] Value extraPart =
        tensor::ExtractSliceOp::create(b, adaptor.getValue(), extraPartOffsets,
                                       extraPartSizes, extraPartStrides);

    // TODO(#2157): implement the rest

    return success();
  }
};

void PolynomialToModArith::runOnOperation() {
  MLIRContext* context = &getContext();

  WalkResult result =
      getOperation()->walk([&](Operation* op) {
        for (Type ty :
             llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
          if (auto polyTy = dyn_cast<PolynomialType>(ty)) {
            if (!polyTy.getRing().getPolynomialModulus()) {
              op->emitError()
                  << "polynomial-to-mod-arith requires all polynomial "
                     "types have a polynomialModulus attribute, but found "
                  << polyTy;
              return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // generateOpImplementations must be called before the conversion begins to
  // apply rewrite patterns, because adding function implementations makes
  // changes at the module level, while the conversion patterns are supposed to
  // be local to the op being converted. This design is borrowed from the MLIR
  // --math-to-funcs pass implementation.
  generateOpImplementations();
  auto getDivmodOp = [&](FunctionType funcType, RingAttr ring) -> func::FuncOp {
    auto it = modImpls.find(std::pair(funcType, ring));
    if (it == modImpls.end()) return nullptr;
    return it->second;
  };

  ModuleOp module = getOperation();
  ConversionTarget target(*context);
  PolynomialToModArithTypeConverter typeConverter(context);

  target.addIllegalDialect<PolynomialDialect>();
  RewritePatternSet patterns(context);

  patterns.add<ConvertFromTensor, ConvertToTensor,
               ConvertPolyBinop<AddOp, arith::AddIOp, mod_arith::AddOp>,
               ConvertPolyBinop<SubOp, arith::SubIOp, mod_arith::SubOp>,
               ConvertLeadingTerm, ConvertMonomial, ConvertMonicMonomialMul,
               ConvertConstant, ConvertMulScalar, ConvertNTT, ConvertINTT,
               ConvertKeySwitchInner>(typeConverter, context);
  patterns.add<ConvertMul>(typeConverter, patterns.getContext(), getDivmodOp);
  addStructuralConversionPatterns(typeConverter, patterns, target);
  addTensorOfTensorConversionPatterns(typeConverter, patterns, target);

  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(module, target, std::move(patterns),
                                    config))) {
    signalPassFailure();
  }
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
