#include "lib/Transforms/SoftmaxToNsSoftmax/SoftmaxToNsSoftmax.h"

#include <cmath>
#include <cstdint>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Transforms/PropagatePadding/PropagatePadding.h"
#include "lib/Utils/Approximation/CaratheodoryFejer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "softmax-to-ns-softmax"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SOFTMAXTONSSOFTMAX
#include "lib/Transforms/SoftmaxToNsSoftmax/SoftmaxToNsSoftmax.h.inc"

namespace {

constexpr double kDefaultDomainLower = -9.0;
constexpr double kDefaultDomainUpper = 6.0;

double f64AttrOr(Operation* op, StringRef name, double dflt) {
  if (auto attr = op->getAttrOfType<FloatAttr>(name))
    return attr.getValueAsDouble();
  return dflt;
}

int32_t i32AttrOr(Operation* op, StringRef name, int32_t dflt) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name)) return attr.getInt();
  return dflt;
}

}  // namespace

struct SoftmaxToNsSoftmax
    : public impl::SoftmaxToNsSoftmaxBase<SoftmaxToNsSoftmax> {
  using SoftmaxToNsSoftmaxBase::SoftmaxToNsSoftmaxBase;

  void runOnOperation() override {
    SmallVector<math_ext::SoftmaxOp> worklist;
    getOperation()->walk(
        [&](math_ext::SoftmaxOp op) { worklist.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (auto op : worklist) {
      if (failed(decompose(rewriter, op))) return signalPassFailure();
    }
  }

 private:
  LogicalResult decompose(IRRewriter& rewriter, math_ext::SoftmaxOp op) {
    auto scoresType = dyn_cast<RankedTensorType>(op.getValue().getType());
    if (!scoresType || !scoresType.hasStaticShape() ||
        !isa<FloatType>(scoresType.getElementType()))
      return op.emitError() << "expected a static float tensor input";

    tensor_ext::PaddingAttr padding = getPaddingInfo(op.getValue());
    bool usePadding = padding && padding.isZeroPadded() &&
                      padding.padded() == scoresType.getShape();

    int64_t n =
        usePadding ? padding.logical().back() : scoresType.getShape().back();
    if (n < 2) return op.emitError() << "softmax row length must be >= 2";

    double domainUpper = f64AttrOr(op, "domain_upper", kDefaultDomainUpper);
    double domainLower = f64AttrOr(op, "domain_lower", kDefaultDomainLower);
    int32_t degree = i32AttrOr(op, "exp_degree", expDegree);
    double pow2k = static_cast<double>(1 << nsK);

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value scores = op.getValue();
    Type elementType = scoresType.getElementType();

    auto shiftAttr = DenseElementsAttr::get(
        scoresType, rewriter.getFloatAttr(elementType, domainUpper));
    Value shiftCst =
        arith::ConstantOp::create(rewriter, loc, scoresType, shiftAttr);
    Value shifted = arith::SubFOp::create(rewriter, loc, scores, shiftCst);
    auto scaleAttr = DenseElementsAttr::get(
        scoresType, rewriter.getFloatAttr(elementType, 1.0 / pow2k));
    Value scaleCst =
        arith::ConstantOp::create(rewriter, loc, scoresType, scaleAttr);
    Value scaled = arith::MulFOp::create(rewriter, loc, shifted, scaleCst);

    Value y = math::ExpOp::create(rewriter, loc, scaled);
    y.getDefiningOp()->setAttr("degree", rewriter.getI32IntegerAttr(degree));
    y.getDefiningOp()->setAttr(
        "domain_lower",
        rewriter.getF64FloatAttr((domainLower - domainUpper) / pow2k));
    y.getDefiningOp()->setAttr("domain_upper", rewriter.getF64FloatAttr(0.0));
    // NS demands near-uniform relative error over the WHOLE stamped
    // interval (renormalization cancels only the uniform part, and the
    // squarings amplify the variation 2^k-fold): pin the Chebyshev/CF
    // solver so the Taylor-by-squaring exp pattern (accurate only near
    // 0; -27.5% at t = -3 with k=4) cannot claim this op. This also
    // keeps the pad-cancellation constant below consistent with the
    // actual circuit polynomial.
    y.getDefiningOp()->setAttr("approximation_method",
                               rewriter.getStringAttr("chebyshev"));

    Value pin;
    if (usePadding) {
      double expLo = (domainLower - domainUpper) / pow2k;
      double padPoint = -domainUpper / pow2k;
      if (domainLower <= 0.0) {
        polynomial::ChebyshevPolynomial poly =
            approximation::caratheodoryFejerApproximation(
                [](const APFloat& x) {
                  return APFloat(std::exp(x.convertToDouble()));
                },
                degree, expLo, 0.0);
        double t = (2.0 * padPoint - (expLo + 0.0)) / (0.0 - expLo);
        double b1 = 0.0, b2 = 0.0;
        ArrayRef<APFloat> terms = poly.getTerms();
        for (int i = static_cast<int>(terms.size()) - 1; i >= 0; --i) {
          double b0 = 2.0 * t * b1 - b2 + terms[i].convertToDouble();
          b2 = b1;
          b1 = b0;
        }
        double padConstant = b1 - t * b2;
        auto corrAttr =
            buildRegionIndicator(scoresType, padding.logical(),
                                 /*onesOutside=*/true, -padConstant);
        Value corrCst =
            arith::ConstantOp::create(rewriter, loc, scoresType, corrAttr);
        y = arith::AddFOp::create(rewriter, loc, y, corrCst);
      } else {
        auto maskAttr = buildRegionIndicator(scoresType, padding.logical(),
                                             /*onesOutside=*/false);
        Value maskCst =
            arith::ConstantOp::create(rewriter, loc, scoresType, maskAttr);
        y = arith::MulFOp::create(rewriter, loc, y, maskCst);
      }

      SmallVector<int64_t> reducedShape(scoresType.getShape().drop_back());
      auto reducedType = RankedTensorType::get(reducedShape, elementType);
      auto pinAttr =
          buildRegionIndicator(reducedType, padding.logical().drop_back(),
                               /*onesOutside=*/true);
      pin = arith::ConstantOp::create(rewriter, loc, reducedType, pinAttr);
    }

    for (int t = 0; t < nsK; ++t) {
      Value sq = arith::MulFOp::create(rewriter, loc, y, y);
      bool last = (t == nsK - 1);
      double lo = last ? 1.0 / (2.0 * n) : 1.0 / (4.0 * n);
      double hi = last ? 1.25 : 2.0 * n;
      y = normalizeByRowSum(rewriter, loc, sq, lo, hi, pin);
    }
    rewriter.replaceOp(op, y);
    return success();
  }

  Value normalizeByRowSum(IRRewriter& rewriter, Location loc, Value x,
                          double recipLo, double recipHi, Value pin) {
    auto type = cast<RankedTensorType>(x.getType());
    int64_t rank = type.getRank();
    Type elementType = type.getElementType();
    SmallVector<int64_t> reducedShape(type.getShape().drop_back());
    auto reducedType = RankedTensorType::get(reducedShape, elementType);
    auto zeroAttr = DenseElementsAttr::get(
        reducedType, rewriter.getFloatAttr(elementType, 0.0));
    Value zeroInit =
        arith::ConstantOp::create(rewriter, loc, reducedType, zeroAttr);
    auto reduce = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{x}, ValueRange{zeroInit},
        ArrayRef<int64_t>{rank - 1},
        [](OpBuilder& b, Location nestedLoc, ValueRange args) {
          Value sum = arith::AddFOp::create(b, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(b, nestedLoc, sum);
        });
    Value sum = reduce.getResult(0);
    if (pin) sum = arith::AddFOp::create(rewriter, loc, sum, pin);
    Value recip = math_ext::RecipOp::create(rewriter, loc, sum);
    recip.getDefiningOp()->setAttr("domain_lower",
                                   rewriter.getF64FloatAttr(recipLo));
    recip.getDefiningOp()->setAttr("domain_upper",
                                   rewriter.getF64FloatAttr(recipHi));
    auto emptyOp =
        tensor::EmptyOp::create(rewriter, loc, type.getShape(), elementType);
    auto bcast = linalg::BroadcastOp::create(
        rewriter, loc, recip, emptyOp.getResult(), ArrayRef<int64_t>{rank - 1});
    return arith::MulFOp::create(rewriter, loc, x, bcast.getResults()[0]);
  }
};

}  // namespace heir
}  // namespace mlir
