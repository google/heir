#include "lib/Transforms/SoftmaxCanonicalizations/SoftmaxCanonicalizations.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SOFTMAXCANONICALIZATIONS
#include "lib/Transforms/SoftmaxCanonicalizations/SoftmaxCanonicalizations.h.inc"

// Matches a linalg.broadcast of `reduced` that exclusively inserts the last
// dimension.
bool matchLastDimBroadcast(Value v, Value& reduced) {
  auto bcast = v.getDefiningOp<linalg::BroadcastOp>();
  if (!bcast) return false;

  auto outType = dyn_cast<RankedTensorType>(v.getType());
  if (!outType) return false;

  ArrayRef<int64_t> dims = bcast.getDimensions();
  if (dims.size() != 1 || dims[0] != outType.getRank() - 1) return false;

  reduced = bcast.getInput();
  return true;
}

// Matches a linalg.reduce over the last dimension of `summed` that sums it
// up, crucially demanding it is initialized to exactly 0.0.
bool matchLastDimZeroInitSum(Value v, Value summed) {
  auto reduceOp = v.getDefiningOp<linalg::ReduceOp>();
  if (!reduceOp || reduceOp.getInputs().size() != 1 ||
      reduceOp.getDpsInits().size() != 1) {
    return false;
  }
  if (reduceOp.getInputs()[0] != summed) return false;

  auto dims = reduceOp.getDimensions();
  auto type = dyn_cast<RankedTensorType>(summed.getType());
  if (!type || dims.size() != 1 || dims[0] != type.getRank() - 1) return false;

  // Verify the reduction init is 0.
  auto cstOp = reduceOp.getDpsInits()[0].getDefiningOp<arith::ConstantOp>();
  if (!cstOp) return false;

  auto denseAttr = dyn_cast<DenseFPElementsAttr>(cstOp.getValue());
  if (!denseAttr || !denseAttr.isSplat() ||
      !denseAttr.getSplatValue<APFloat>().isZero()) {
    return false;
  }

  // Verify it's an addition reduction.
  Block& body = reduceOp.getCombiner().front();
  if (body.getOperations().size() != 2) return false;

  auto addOp = dyn_cast<arith::AddFOp>(body.front());
  if (!addOp || addOp.getLhs() != body.getArgument(0) ||
      addOp.getRhs() != body.getArgument(1)) {
    return false;
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.back());
  if (!yieldOp || yieldOp.getOperand(0) != addOp.getResult()) return false;

  return true;
}

// Matches a linalg.reduce over the last dimension of `scores` that takes the
// maximum, demanding it is initialized to exactly -inf.
bool matchMaxReduction(Value scores, Value maxVal) {
  auto genericOp = maxVal.getDefiningOp<linalg::GenericOp>();
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getDpsInputOperand(0)->get() != scores) {
    return false;
  }

  // 1. Identify which output 'maxVal' correlates to.
  auto resultIt = llvm::find(genericOp.getResults(), maxVal);
  if (resultIt == genericOp.getResults().end()) return false;
  int resultIdx = std::distance(genericOp.getResults().begin(), resultIt);

  // 2. Check that its corresponding init value is a negative infinity splat.
  Value initVal = genericOp.getDpsInitOperand(resultIdx)->get();
  auto cstOp = initVal.getDefiningOp<arith::ConstantOp>();
  if (!cstOp) return false;
  auto denseAttr = dyn_cast<DenseFPElementsAttr>(cstOp.getValue());
  if (!denseAttr || !denseAttr.isSplat() ||
      !denseAttr.getSplatValue<APFloat>().isInfinity() ||
      !denseAttr.getSplatValue<APFloat>().isNegative()) {
    return false;
  }

  // 3. Check the iterator types (last dim is reduction, others are parallel).
  auto iterTypes = genericOp.getIteratorTypesArray();
  if (iterTypes.empty() || iterTypes.back() != utils::IteratorType::reduction)
    return false;
  for (size_t i = 0; i < iterTypes.size() - 1; ++i) {
    if (iterTypes[i] != utils::IteratorType::parallel) return false;
  }

  // 4. Verify the operation body rigorously.
  Block& body = genericOp.getRegion().front();
  Value inputArg = body.getArgument(0);
  Value accArg = body.getArgument(1 + resultIdx);  // Inits follow inputs

  auto yieldOp = cast<linalg::YieldOp>(body.back());
  Value yieldedMax = yieldOp.getOperand(resultIdx);

  auto maxOp = yieldedMax.getDefiningOp();
  if (!maxOp) return false;

  if (!isa<arith::MaximumFOp, arith::MaxNumFOp>(maxOp)) {
    return false;
  }

  Value lhs = maxOp->getOperand(0);
  Value rhs = maxOp->getOperand(1);
  if (!((lhs == inputArg && rhs == accArg) ||
        (rhs == inputArg && lhs == accArg))) {
    return false;
  }

  return true;
}

// Looks for and strips the optional shift-invariance stabilization:
// `arith.subf(scores, broadcast(...))`
Value stripMaxSubtraction(Value expInput) {
  auto subOp = expInput.getDefiningOp<arith::SubFOp>();
  if (!subOp) return expInput;

  Value scores = subOp.getLhs();
  Value maxBcast = subOp.getRhs();

  Value maxVal;
  if (!matchLastDimBroadcast(maxBcast, maxVal)) return expInput;

  // Verify maxVal comes from a rigorous max reduction of `scores`
  if (!matchMaxReduction(scores, maxVal)) return expInput;

  return scores;
}

struct RaiseTorchSoftmaxPattern : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter& rewriter) const override {
    // 0. DivFOp must operate on ranked tensors. Scalar divisions
    // are instantly ignored.
    if (!isa<RankedTensorType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "not a ranked tensor division");
    }

    // 1. The numerator must be the result of a math.exp operation. This is
    // checked first since exponentiation is much more rare than broadcasting,
    // cutting off non-softmax divisions (like standardizations) quickly.
    auto expOp = op.getLhs().getDefiningOp<math::ExpOp>();
    if (!expOp) {
      return rewriter.notifyMatchFailure(op, "numerator is not math.exp");
    }

    // 2. The denominator must be a last-dim broadcast.
    Value sumVal;
    if (!matchLastDimBroadcast(op.getRhs(), sumVal)) {
      return rewriter.notifyMatchFailure(
          op, "denominator is not a last-dim broadcast");
    }

    Value numerator = op.getLhs();

    // 3. Denominator must be a zero-init sum of the exponential
    if (!matchLastDimZeroInitSum(sumVal, numerator)) {
      return rewriter.notifyMatchFailure(
          op, "denominator is not a zero-init sum of the exp");
    }

    // 4. Optionally strip the max-subtraction stabilization
    Value scores = stripMaxSubtraction(expOp.getOperand());

    // 5. Final shape/rank assertions
    auto scoresType = dyn_cast<RankedTensorType>(scores.getType());
    if (!scoresType || !scoresType.hasStaticShape() ||
        !isa<FloatType>(scoresType.getElementType()) ||
        scoresType.getRank() < 1 || scoresType.getRank() > 3) {
      return rewriter.notifyMatchFailure(
          op, "scores must be a statically shaped float tensor of rank 1-3");
    }

    // 6. Rewrite to math_ext.softmax
    auto softmax =
        math_ext::SoftmaxOp::create(rewriter, op.getLoc(), scoresType, scores);
    softmax->setAttr("dimension",
                     rewriter.getI64IntegerAttr(scoresType.getRank() - 1));
    rewriter.replaceOp(op, softmax.getResult());
    return success();
  }
};

struct SoftmaxCanonicalizations
    : public impl::SoftmaxCanonicalizationsBase<SoftmaxCanonicalizations> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseTorchSoftmaxPattern>(&getContext());
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
