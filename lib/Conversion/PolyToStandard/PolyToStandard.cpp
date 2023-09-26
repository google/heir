#include "include/Conversion/PolyToStandard/PolyToStandard.h"

#include "include/Dialect/Poly/IR/PolyOps.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "lib/Conversion/Utils.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "include/Conversion/PolyToStandard/PolyToStandard.h.inc"

// Callback type for getting pre-generated FuncOp implementing
// helper functions for various lowerings.
using GetFuncCallbackTy = function_ref<func::FuncOp(FunctionType, RingAttr)>;

RankedTensorType convertPolyType(PolyType type) {
  RingAttr attr = type.getRing();
  uint32_t idealDegree = attr.ideal().getDegree();
  // We subtract one because the maximum value of a coefficient is one less
  // than the modulus. When the modulus is an exact power of 2 this matters.
  unsigned eltBitWidth = (attr.coefficientModulus() - 1).getActiveBits();
  IntegerType elementTy =
      IntegerType::get(type.getContext(), eltBitWidth,
                       IntegerType::SignednessSemantics::Signless);
  // We must remove the ring attribute on the tensor, since the
  // unrealized_conversion_casts cannot carry the poly.ring attribute
  // through.
  return RankedTensorType::get({idealDegree}, elementTy);
}

class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PolyType type) -> Type { return convertPolyType(type); });

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
  ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultTy = typeConverter->convertType(op->getResultTypes()[0]);
    auto resultTensorTy = cast<RankedTensorType>(resultTy);
    auto resultShape = resultTensorTy.getShape()[0];
    auto resultEltTy = resultTensorTy.getElementType();

    auto inputTensorTy = op.getInput().getType();
    auto inputShape = inputTensorTy.getShape()[0];
    auto inputEltTy = inputTensorTy.getElementType();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getInput();
    // Extend element type if needed.
    if (inputEltTy != resultEltTy) {
      // FromTensorOp verifies that the coefficient tensor's elements fit into
      // the polynomial.
      assert(inputEltTy.getIntOrFloatBitWidth() <
             resultEltTy.getIntOrFloatBitWidth());

      coeffValue = b.create<arith::ExtUIOp>(
          RankedTensorType::get(inputShape, resultEltTy), coeffValue);
    }

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(resultShape - inputShape));
      coeffValue = b.create<tensor::PadOp>(
          resultTy, coeffValue, low, high,
          b.create<arith::ConstantOp>(rewriter.getIntegerAttr(resultEltTy, 0)),
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
    RankedTensorType tensorType = cast<RankedTensorType>(
        typeConverter->convertType(op.getResult().getType()));
    PolynomialAttr attr = op.getInput();
    SmallVector<Attribute> coeffs;
    auto eltTy = tensorType.getElementType();
    unsigned numTerms = tensorType.getShape()[0];
    coeffs.reserve(numTerms);
    // This is inefficient for large-degree polys, but as of this writing we
    // don't have a lowering that uses a sparse representation.
    for (size_t i = 0; i < numTerms; ++i) {
      coeffs.push_back(rewriter.getIntegerAttr(eltTy, 0));
    }
    for (const auto &term : attr.getPolynomial().getTerms()) {
      coeffs[term.exponent.getZExtValue()] = rewriter.getIntegerAttr(
          eltTy, term.coefficient.trunc(eltTy.getIntOrFloatBitWidth()));
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, DenseElementsAttr::get(tensorType, coeffs));
    return success();
  }
};

struct ConvertMonomial : public OpConversionPattern<MonomialOp> {
  ConvertMonomial(mlir::MLIRContext *context)
      : OpConversionPattern<MonomialOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MonomialOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto polyType = cast<PolyType>(op.getResult().getType());
    auto tensorType =
        cast<RankedTensorType>(typeConverter->convertType(polyType));
    auto tensor = b.create<arith::ConstantOp>(DenseElementsAttr::get(
        tensorType, b.getIntegerAttr(tensorType.getElementType(), 0)));
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getCoefficient(),
                                                  tensor, adaptor.getDegree());
    return success();
  }
};

struct ConvertMulScalar : public OpConversionPattern<MulScalarOp> {
  ConvertMulScalar(mlir::MLIRContext *context)
      : OpConversionPattern<MulScalarOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulScalarOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensorType = cast<RankedTensorType>(adaptor.getPolynomial().getType());
    Value scalar = adaptor.getScalar();
    if (scalar.getType() != tensorType.getElementType()) {
      scalar = b.create<arith::ExtUIOp>(tensorType.getElementType(), scalar)
                   .getResult();
    }
    auto tensor = b.create<tensor::SplatOp>(tensorType, scalar);
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getPolynomial(),
                                               tensor);
    return success();
  }
};

// Implement rotation via tensor.insert_slice
struct ConvertMonomialMul : public OpConversionPattern<MonomialMulOp> {
  ConvertMonomialMul(mlir::MLIRContext *context)
      : OpConversionPattern<MonomialMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MonomialMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // In general, a rotation would correspond to multiplication by x^n,
    // which requires a modular reduction step. But because the verifier
    // requires the ring to have a specific structure (x^n - 1), this op
    // can be implemented as a cyclic shift with wraparound.
    auto tensorType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputTensorContainer = b.create<tensor::EmptyOp>(
        tensorType.getShape(), tensorType.getElementType());

    RankedTensorType dynamicSliceType = RankedTensorType::get(
        ShapedType::kDynamic, tensorType.getElementType());

    // split the tensor into two pieces at index N - rotation_amount
    // e.g., if rotation_amount is 2,
    //
    //   [0, 1, 2, 3, 4 | 5, 6]
    //
    // the resulting output is
    //
    //   [5, 6 | 0, 1, 2, 3, 4]
    auto constTensorDim = b.create<arith::ConstantOp>(
        b.getIndexType(), b.getIndexAttr(tensorType.getShape()[0]));
    auto splitPoint =
        b.create<arith::SubIOp>(constTensorDim, adaptor.getMonomialDegree());

    SmallVector<OpFoldResult> firstHalfExtractOffsets{b.getIndexAttr(0)};
    SmallVector<OpFoldResult> firstHalfExtractSizes{splitPoint.getResult()};
    SmallVector<OpFoldResult> strides{b.getIndexAttr(1)};
    auto firstHalfExtractOp = b.create<tensor::ExtractSliceOp>(
        /*resultType=*/dynamicSliceType,
        /*source=*/adaptor.getInput(), firstHalfExtractOffsets,
        firstHalfExtractSizes, strides);

    SmallVector<OpFoldResult> secondHalfExtractOffsets{splitPoint.getResult()};
    SmallVector<OpFoldResult> secondHalfExtractSizes{
        adaptor.getMonomialDegree()};
    auto secondHalfExtractOp = b.create<tensor::ExtractSliceOp>(
        /*resultType=*/dynamicSliceType,
        /*source=*/adaptor.getInput(), secondHalfExtractOffsets,
        secondHalfExtractSizes, strides);

    SmallVector<OpFoldResult> firstHalfInsertOffsets{
        adaptor.getMonomialDegree()};
    SmallVector<OpFoldResult> firstHalfInsertSizes{splitPoint.getResult()};
    auto firstHalfInsertOp = b.create<tensor::InsertSliceOp>(
        /*source=*/firstHalfExtractOp.getResult(),
        /*dest=*/outputTensorContainer.getResult(), firstHalfInsertOffsets,
        firstHalfInsertSizes, strides);

    SmallVector<OpFoldResult> secondHalfInsertOffsets{b.getIndexAttr(0)};
    SmallVector<OpFoldResult> secondHalfInsertSizes{
        adaptor.getMonomialDegree()};
    auto secondHalfInsertOp = b.create<tensor::InsertSliceOp>(
        /*source=*/secondHalfExtractOp.getResult(),
        /*dest=*/firstHalfInsertOp.getResult(), secondHalfInsertOffsets,
        secondHalfInsertSizes, strides);

    rewriter.replaceOp(op, secondHalfInsertOp.getResult());
    return success();
  }
};

struct ConvertLeadingTerm : public OpConversionPattern<LeadingTermOp> {
  ConvertLeadingTerm(mlir::MLIRContext *context)
      : OpConversionPattern<LeadingTermOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LeadingTermOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffs = adaptor.getInput();
    auto tensorType = cast<RankedTensorType>(coeffs.getType());
    auto c0 = b.create<arith::ConstantOp>(
        b.getIntegerAttr(tensorType.getElementType(), 0));
    auto c1 = b.create<arith::ConstantOp>(b.getIndexAttr(1));
    auto initIndex = b.create<arith::ConstantOp>(
        b.getIndexAttr(tensorType.getShape()[0] - 1));

    auto degreeOp = b.create<scf::WhileOp>(
        /*resultTypes=*/
        TypeRange{b.getIndexType()},
        /*operands=*/ValueRange{initIndex.getResult()},
        /*beforeBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value index = args[0];
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          auto coeff = b.create<tensor::ExtractOp>(coeffs, ValueRange{index});
          auto cmpOp =
              b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, coeff, c0);
          b.create<scf::ConditionOp>(cmpOp.getResult(), index);
        },
        /*afterBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          Value currentIndex = args[0];
          auto nextIndex =
              b.create<arith::SubIOp>(currentIndex, c1.getResult());
          b.create<scf::YieldOp>(nextIndex.getResult());
        });
    auto degree = degreeOp.getResult(0);
    auto leadingCoefficient =
        b.create<tensor::ExtractOp>(coeffs, ValueRange{degree});
    rewriter.replaceOp(op, ValueRange{degree, leadingCoefficient.getResult()});
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Convert add lowers a poly.add operation to arith operations. A poly.add
  // operation is defined within the polynomial ring. Coefficients are added
  // element-wise as elements of the ring, so they are performed modulo the
  // coefficient modulus.
  //
  // This lowering detects when the ring's coefficient modulus is a power of 2,
  // and hence the natural overflow semantics can be relied upon to implement
  // modular arithmetic. In other cases, explicit modular arithmetic operations
  // are inserted, which requires representing the modulus as a constant, and
  // hence may require extending the intermediate arithmetic to higher bit
  // widths.
  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto type = cast<ShapedType>(adaptor.getLhs().getType());

    APInt mod =
        cast<PolyType>(op.getResult().getType()).getRing().coefficientModulus();
    bool needToExtend = !mod.isPowerOf2();

    if (!needToExtend) {
      auto result = b.create<arith::AddIOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, result);
      return success();
    }

    // The arithmetic may spill into higher bit width, so start by extending
    // all the types to the smallest bit width that can contain them all.
    unsigned nextHigherBitWidth = (mod - 1).getActiveBits() + 1;
    auto modIntType = rewriter.getIntegerType(nextHigherBitWidth);
    auto modIntTensorType = RankedTensorType::get(type.getShape(), modIntType);

    auto cmod = b.create<arith::ConstantOp>(DenseIntElementsAttr::get(
        modIntTensorType, {mod.trunc(nextHigherBitWidth)}));

    auto signExtensionLhs =
        b.create<arith::ExtUIOp>(modIntTensorType, adaptor.getLhs());
    auto signExtensionRhs =
        b.create<arith::ExtUIOp>(modIntTensorType, adaptor.getRhs());

    auto higherBitwidthAdd =
        b.create<arith::AddIOp>(signExtensionLhs, signExtensionRhs);
    auto remOp = b.create<arith::RemUIOp>(higherBitwidthAdd, cmod);
    auto truncOp = b.create<arith::TruncIOp>(type, remOp);

    rewriter.replaceOp(op, truncOp);

    return success();
  }
};

unsigned nextPow2(unsigned x) {
  return x == 1 ? 1 : 1 << (32 - APInt(32, x - 1).countLeadingZeros());
}

RankedTensorType naivePolymulOutputTensorType(PolyType type) {
  auto convNumBits =
      (type.getRing().coefficientModulus() - 1).getActiveBits() * 2;
  auto convNumBitsRounded = nextPow2(convNumBits);
  auto eltType = IntegerType::get(type.getContext(), convNumBitsRounded);
  auto convDegree = 2 * type.getRing().getIdeal().getDegree() - 1;
  return RankedTensorType::get({convDegree}, eltType);
}

// Lower polynomial multiplication to a 1D convolution, followed by with a
// modulus reduction in the ring.
struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(const TypeConverter &typeConverter, mlir::MLIRContext *context,
             GetFuncCallbackTy cb)
      : OpConversionPattern<MulOp>(typeConverter, context),
        getFuncOpCallback(cb) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // TODO(https://github.com/google/heir/issues/143): Handle tensor of polys.
    auto polyTy = dyn_cast<PolyType>(op.getResult().getType());
    if (!polyTy) {
      return failure();
    }

    // Implementing a naive polymul operation which is a loop
    //
    // for i = 0, ..., N-1
    //   for j = 0, ..., N-1
    //     c[i+j] += a[i] * b[j]
    //
    RankedTensorType polymulTensorType = naivePolymulOutputTensorType(polyTy);
    auto inputOutputHaveSameBitwidth =
        polymulTensorType.getElementType().getIntOrFloatBitWidth() ==
        dyn_cast<RankedTensorType>(adaptor.getLhs().getType())
            .getElementType()
            .getIntOrFloatBitWidth();
    SmallVector<utils::IteratorType> iteratorTypes(
        2, utils::IteratorType::parallel);
    AffineExpr d0, d1;
    bindDims(getContext(), d0, d1);
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(2, 0, {d0}),      // i
        AffineMap::get(2, 0, {d1}),      // j
        AffineMap::get(2, 0, {d0 + d1})  // i+j
    };

    // TODO(https://github.com/google/heir/issues/200): test this for the proper
    // setting of the DenseElementsAttr. Note that this fails an MLIR internal
    // assertion if 0L (64 bit) is replaced with 0 (32 bit). Probably it is best
    // to inspect the element type of poylmulTensorType and figure out how to
    // set the dense elements attr properly based on that (i.e., what if it's an
    // i16 or i128?)
    auto polymulOutput = b.create<arith::ConstantOp>(
        polymulTensorType, DenseElementsAttr::get(polymulTensorType, 0L));
    auto polyMul = b.create<linalg::GenericOp>(
        /*resultTypes=*/polymulTensorType,
        /*inputs=*/adaptor.getOperands(),
        /*outputs=*/polymulOutput.getResult(),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          auto lhs = args[0];
          auto rhs = args[1];
          auto accum = args[2];
          if (!inputOutputHaveSameBitwidth) {
            lhs = b.create<arith::ExtUIOp>(polymulTensorType.getElementType(),
                                           lhs);
            rhs = b.create<arith::ExtUIOp>(polymulTensorType.getElementType(),
                                           rhs);
          }
          auto mulOp = b.create<arith::MulIOp>(lhs, rhs);
          auto result = b.create<arith::AddIOp>(mulOp, accum);
          // TODO(https://github.com/google/heir/issues/201): need a modular
          // reduction op here to keep all the arithmetic mod cmod.
          // And simplify the code here and in the poly mod func to use smaller
          // bit widths.
          b.create<linalg::YieldOp>(result.getResult());
        });

    auto postReductionType = convertPolyType(polyTy);
    FunctionType funcType = FunctionType::get(
        op.getContext(), {polymulTensorType}, {postReductionType});

    // 2N - 1 sized result tensor -> reduce modulo ideal to get a N sized tensor
    func::FuncOp divMod = getFuncOpCallback(funcType, polyTy.getRing());
    if (!divMod)
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Missing software implementation for polynomial mod op of type"
             << funcType << " and for ring " << polyTy.getRing();
      });

    rewriter.replaceOpWithNewOp<func::CallOp>(op, divMod, polyMul.getResult(0));
    return success();
  }

 private:
  GetFuncCallbackTy getFuncOpCallback;
};

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override;

 private:
  // Generate implementations for operations
  void generateOpImplementations();

  func::FuncOp buildPolyModFunc(FunctionType funcType, RingAttr ringAttr);

  // A map containing modular reduction function implementations, generated once
  // at the beginning of this pass based on the ops to be converted, intended to
  // be retrieved by ConvertMul to construct CallOps so that later optimization
  // passes can determine when to inline the implementation.
  DenseMap<std::pair<Type, RingAttr>, func::FuncOp> modImpls;
};

void PolyToStandard::generateOpImplementations() {
  ModuleOp module = getOperation();
  module.walk([&](MulOp op) {
    auto polyTy = dyn_cast<PolyType>(op.getResult().getType());
    if (!polyTy) {
      // TODO(https://github.com/google/heir/issues/143): Handle tensor of
      // polys.
      return WalkResult::interrupt();
    }
    auto convType = naivePolymulOutputTensorType(polyTy);
    auto postReductionType = convertPolyType(polyTy);
    FunctionType funcType =
        FunctionType::get(op.getContext(), {convType}, {postReductionType});

    // Generate the software implementation of modular reduction if it has not
    // been generated yet.
    auto key = std::pair(funcType, polyTy.getRing());
    if (!modImpls.count(key)) {
      func::FuncOp modOp = buildPolyModFunc(funcType, polyTy.getRing());
      modImpls.insert(std::pair(key, modOp));
    }
    return WalkResult::advance();
  });
}

// Create a software implementation that reduces a polynomial
// modulo a statically known divisor polynomial.
func::FuncOp PolyToStandard::buildPolyModFunc(FunctionType funcType,
                                              RingAttr ring) {
  ModuleOp module = getOperation();
  Location loc = module->getLoc();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  // These tensor types are used in the implementation
  //
  //  - The input tensor<2047xi64>, e.g., the output of a naive polymul of two
  //    tensor<1024xi32>
  //  - The result tensor<1024xi32>, e.g., the result after modular reduction is
  //    complete
  //  - An intermediate tensor<1024xi64>, representing the remainder being
  //    accumulated, before a final remui op is applied to its coefficients.
  RankedTensorType inputType =
      llvm::cast<RankedTensorType>(funcType.getInput(0));
  RankedTensorType resultType =
      llvm::cast<RankedTensorType>(funcType.getResult(0));

  // TODO(https://github.com/google/heir/issues/202): this function name
  // probably also needs the input tensor type in the name, or it could conflict
  // with other implementations that have the same cmod+ideal.
  std::string funcName =
      llvm::formatv("__heir_poly_mod_{0}_{1}", ring.coefficientModulus(),
                    ring.getIdeal().toIdentifier());

  auto funcOp = builder.create<func::FuncOp>(funcName, funcType);
  LLVM::linkage::Linkage inlineLinkage = LLVM::linkage::Linkage::LinkonceODR;
  Attribute linkage =
      LLVM::LinkageAttr::get(builder.getContext(), inlineLinkage);
  funcOp->setAttr("llvm.linkage", linkage);
  funcOp.setPrivate();

  Block *funcBody = funcOp.addEntryBlock();
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
  // ring of polynomials mod (x^n - 1) with sufficiently large coefficient
  // modulus to encapsulate the input element type.

  std::vector<Monomial> monomials;
  // If the input has size N as a tensor, then as a polynomial its max degree is
  // N-1, and we want the ring to be mod (x^N - 1).
  unsigned remRingDegree = inputType.getShape()[0];
  monomials.emplace_back(1, remRingDegree);
  monomials.emplace_back(Monomial(-1, 0));
  Polynomial xnMinusOne = Polynomial::fromMonomials(monomials, &getContext());
  // e.g., need to represent 2^64, which requires 65 bits, the highest one set.
  unsigned remCmodWidth =
      1 + inputType.getElementType().getIntOrFloatBitWidth();
  APInt remCmod = APInt::getOneBitSet(remCmodWidth, remCmodWidth - 1);
  auto remRing = RingAttr::get(remCmod, xnMinusOne);
  auto remRingPolyType = PolyType::get(&getContext(), remRing);

  // Start by converting the input tensor back to a poly.
  auto fromTensorOp = builder.create<FromTensorOp>(coeffsArg, remRing);

  // If the leading coefficient of the divisor has no inverse, we can't do
  // division. The lowering must fail:
  auto divisor = ring.getIdeal();
  auto leadingCoef = divisor.getTerms().back().coefficient;
  auto leadingCoefInverse = leadingCoef.multiplicativeInverse(ring.getCmod());
  // APInt signals no inverse by returning zero.
  if (leadingCoefInverse.isZero()) {
    signalPassFailure();
  }
  auto divisorLcInverse = builder.create<arith::ConstantOp>(
      inputType.getElementType(),
      builder.getIntegerAttr(inputType.getElementType(),
                             leadingCoefInverse.getZExtValue()));

  auto divisorDeg = builder.create<arith::ConstantOp>(
      builder.getIndexType(), builder.getIndexAttr(divisor.getDegree()));

  auto remainder = fromTensorOp.getResult();

  // while remainder.degree() >= divisorDeg:
  auto whileOp = builder.create<scf::WhileOp>(
      /*resultTypes=*/
      remainder.getType(),
      /*operands=*/remainder,
      /*beforeBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value remainder = args[0];
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        // remainder.degree() >= divisorDeg
        auto remainderLt = b.create<LeadingTermOp>(
            b.getIndexType(), inputType.getElementType(), remainder);
        auto cmpOp = b.create<arith::CmpIOp>(
            arith::CmpIPredicate::sge, remainderLt.getDegree(), divisorDeg);
        b.create<scf::ConditionOp>(cmpOp, remainder);
      },
      /*afterBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value remainder = args[0];
        // TODO(https://github.com/google/heir/issues/97): move this out of the
        // loop when it has ConstantLike trait
        auto divisorOp = builder.create<ConstantOp>(
            remRingPolyType, PolynomialAttr::get(divisor));

        // monomialExponent = remainder.degree() - divisorDeg
        auto ltOp = b.create<LeadingTermOp>(
            b.getIndexType(), inputType.getElementType(), remainder);
        auto monomialExponentOp =
            b.create<arith::SubIOp>(ltOp.getDegree(), divisorDeg);

        // monomialDivisor = monomial(
        //   monomialExponent, remainder.leadingCoefficient() / divisorLC)
        auto monomialLc =
            b.create<arith::MulIOp>(ltOp.getCoefficient(), divisorLcInverse);

        // remainder -= monomialDivisor * divisor
        auto scaledDivisor = b.create<MulScalarOp>(divisorOp, monomialLc);
        auto remainderIncrement = b.create<MonomialMulOp>(
            scaledDivisor, monomialExponentOp.getResult());
        auto nextRemainder = b.create<SubOp>(remainder, remainderIncrement);

        b.create<scf::YieldOp>(nextRemainder.getResult());
      });

  // The result remainder is still in the larger ring, but needs to have its
  // coefficients modded and then converted to the smaller ring.
  auto toTensorOp = builder.create<ToTensorOp>(inputType, whileOp.getResult(0));
  auto remainderModOp = builder.create<arith::ConstantOp>(
      inputType,
      DenseElementsAttr::get(inputType, ring.getCmod().getZExtValue()));
  auto remainderCoeffsModded = builder.create<arith::RemUIOp>(
      toTensorOp.getResult(), remainderModOp.getResult());

  // Now the remainder has values mod the result modulus, so it just needs
  // to be reinterpreted in the result ring via truncating the tensor type
  // and changing the element type.
  auto remainderIntegerTrunced = builder.create<arith::TruncIOp>(
      RankedTensorType::get(inputType.getShape(), resultType.getElementType()),
      remainderCoeffsModded);

  SmallVector<OpFoldResult> offsets{builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes{
      builder.getIndexAttr(resultType.getShape()[0])};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
  auto remainderTensorTrunced = builder.create<tensor::ExtractSliceOp>(
      resultType, remainderIntegerTrunced.getResult(), offsets, sizes, strides);

  builder.create<func::ReturnOp>(remainderTensorTrunced.getResult());
  return funcOp;
}

void PolyToStandard::runOnOperation() {
  MLIRContext *context = &getContext();
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
  PolyToStandardTypeConverter typeConverter(context);

  target.addLegalDialect<arith::ArithDialect>();

  // target.addIllegalDialect<PolyDialect>();
  target.addIllegalOp<FromTensorOp, ToTensorOp, AddOp, SubOp, LeadingTermOp,
                      MonomialOp, MulScalarOp, MonomialMulOp, ConstantOp,
                      MulOp>();

  RewritePatternSet patterns(context);

  // Rewrite Sub as Add, to avoid lowering both
  RewritePatternSet canonicalizationPatterns(context);
  SubOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  (void)applyPatternsAndFoldGreedily(module,
                                     std::move(canonicalizationPatterns));

  patterns.add<ConvertFromTensor, ConvertToTensor, ConvertAdd,
               ConvertLeadingTerm, ConvertMonomial, ConvertMonomialMul,
               ConvertConstant, ConvertMulScalar>(typeConverter, context);
  patterns.add<ConvertMul>(typeConverter, patterns.getContext(), getDivmodOp);
  addStructuralConversionPatterns(typeConverter, patterns, target);

  // TODO(https://github.com/google/heir/issues/143): Handle tensor of polys.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace poly
}  // namespace heir
}  // namespace mlir
