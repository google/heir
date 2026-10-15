#include "lib/Transforms/SoftmaxToCgfSoftmax/SoftmaxToCgfSoftmax.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SOFTMAXTOCGFSOFTMAX
#include "lib/Transforms/SoftmaxToCgfSoftmax/SoftmaxToCgfSoftmax.h.inc"

namespace {

// Helper to create a linalg.reduce sum operation.
// Returns the reduced tensor.
struct SoftmaxToCgfSoftmaxPattern
    : public OpRewritePattern<math_ext::SoftmaxOp> {
  using OpRewritePattern<math_ext::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math_ext::SoftmaxOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getValue();
    auto inputType = cast<RankedTensorType>(input.getType());
    assert(inputType.hasStaticShape() && "only static shapes are supported");
    int64_t rank = inputType.getRank();
    assert((rank == 1 || rank == 2) && "only 1D and 2D tensors are supported");

    Type elemType = inputType.getElementType();
    auto inputShape = inputType.getShape();
    int64_t n = inputShape[rank - 1];
    double n_double = static_cast<double>(n);

    Value invNConst = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elemType, 1.0 / n_double));
    Value halfConst = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elemType, 0.5));
    Value lnNConst = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elemType, std::log(n_double)));

    int64_t reductionDim = rank - 1;
    auto reductionDimAttr = rewriter.getI64IntegerAttr(reductionDim);
    auto addfAttr = rewriter.getStringAttr("arith.addf");

    // 1. Compute mean (mu)
    Value sumBroadcast = tensor_ext::BroadcastedReduceOp::create(
        rewriter, loc, inputType, input, reductionDimAttr, addfAttr);
    Value invNConstSplat =
        tensor::SplatOp::create(rewriter, loc, inputType, invNConst);
    Value muBroadcast =
        arith::MulFOp::create(rewriter, loc, sumBroadcast, invNConstSplat);

    // 2. Compute variance (sigma^2)
    Value diff = arith::SubFOp::create(rewriter, loc, input, muBroadcast);
    Value diffSq = arith::MulFOp::create(rewriter, loc, diff, diff);

    Value sumDiffSqBroadcast = tensor_ext::BroadcastedReduceOp::create(
        rewriter, loc, inputType, diffSq, reductionDimAttr, addfAttr);
    Value sigmaSqBroadcast = arith::MulFOp::create(
        rewriter, loc, sumDiffSqBroadcast, invNConstSplat);

    // 3. Compute shift S = mu + sigma_sq / 2 + ln(n)
    Value halfSplat =
        tensor::SplatOp::create(rewriter, loc, inputType, halfConst);
    Value lnNSplat =
        tensor::SplatOp::create(rewriter, loc, inputType, lnNConst);
    Value halfSigmaSqBroadcast =
        arith::MulFOp::create(rewriter, loc, sigmaSqBroadcast, halfSplat);
    Value muPlusHalfSigmaSqBroadcast =
        arith::AddFOp::create(rewriter, loc, muBroadcast, halfSigmaSqBroadcast);
    Value shiftBroadcast = arith::AddFOp::create(
        rewriter, loc, muPlusHalfSigmaSqBroadcast, lnNSplat);

    // 4. Shift inputs and apply exp: result = exp(input - shift)
    double L_val =
        op->hasAttr("domain_lower")
            ? cast<FloatAttr>(op->getAttr("domain_lower")).getValueAsDouble()
            : -1.0;
    double U_val =
        op->hasAttr("domain_upper")
            ? cast<FloatAttr>(op->getAttr("domain_upper")).getValueAsDouble()
            : 1.0;
    double est_lower =
        L_val -
        (U_val + (U_val - L_val) * (U_val - L_val) / 8.0 + std::log(n_double));
    double safe_lower = std::max(est_lower, -16.0);

    Value shiftedInput =
        arith::SubFOp::create(rewriter, loc, input, shiftBroadcast);
    auto expOp = math::ExpOp::create(rewriter, loc, shiftedInput);
    expOp->setAttr("domain_lower", rewriter.getF64FloatAttr(safe_lower));
    expOp->setAttr("domain_upper", rewriter.getF64FloatAttr(0.5));

    rewriter.replaceOp(op, expOp->getResults());
    return success();
  }
};

}  // namespace

struct SoftmaxToCgfSoftmaxPass
    : public impl::SoftmaxToCgfSoftmaxBase<SoftmaxToCgfSoftmaxPass> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    Operation* op = getOperation();

    // Pre-check for errors/warnings on domain width.
    WalkResult walkResult = op->walk([](math_ext::SoftmaxOp softmaxOp) {
      auto lowerAttr =
          dyn_cast_or_null<FloatAttr>(softmaxOp->getAttr("domain_lower"));
      auto upperAttr =
          dyn_cast_or_null<FloatAttr>(softmaxOp->getAttr("domain_upper"));
      if (lowerAttr && upperAttr) {
        double L = lowerAttr.getValueAsDouble();
        double U = upperAttr.getValueAsDouble();
        double width = U - L;
        if (width > 4.0) {
          softmaxOp->emitOpError()
              << "input domain width (" << width
              << ") exceeds the maximum safe limit (4.0) for CGF-softmax "
                 "approximation";
          return WalkResult::interrupt();
        } else if (width > 2.0) {
          softmaxOp->emitWarning()
              << "input domain width (" << width
              << ") exceeds the recommended safe limit (2.0) for CGF-softmax "
                 "approximation. Accuracy may degrade.";
        }
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);
    patterns.add<SoftmaxToCgfSoftmaxPattern>(context);

    walkAndApplyPatterns(op, std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
