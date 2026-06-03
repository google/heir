#include "lib/Transforms/LinalgFuseLinearOps/LinalgFuseLinearOps.h"

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"    // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"    // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LINALGFUSELINEAROPS
#include "lib/Transforms/LinalgFuseLinearOps/LinalgFuseLinearOps.h.inc"

namespace {

template <typename OpTy>
LogicalResult findLinearOpAndOperand(OpTy op, Operation*& linearOp,
                                     Value& rawOperand) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();

  auto defOp = lhs.getDefiningOp();
  Value other = rhs;
  if (!defOp || !isa<linalg::LinalgOp>(defOp)) {
    defOp = rhs.getDefiningOp();
    other = lhs;
  }
  if (!defOp || !isa<linalg::LinalgOp>(defOp)) return failure();

  linearOp = defOp;
  rawOperand = other;
  if (Operation* broadcastOp = other.getDefiningOp()) {
    if (broadcastOp->getName().getStringRef() == "linalg.broadcast") {
      rawOperand = broadcastOp->getOperand(0);
    }
  }
  return success();
}

template <typename OpTy>
LogicalResult fuseScaleOrDivIntoLinearOp(PatternRewriter& rewriter, OpTy op) {
  Operation* linearOp = nullptr;
  Value scale_val;
  if (failed(findLinearOpAndOperand(op, linearOp, scale_val))) return failure();

  Value weights;
  int64_t weightOperandIdx = -1;
  int64_t matchDim = -1;

  llvm::TypeSwitch<Operation*>(linearOp)
      .template Case<linalg::MatmulOp, linalg::VecmatOp>([&](auto op) {
        weights = op.getOperand(1);
        weightOperandIdx = 1;
      })
      .template Case<linalg::MatvecOp>([&](auto op) {
        weights = op.getOperand(0);
        weightOperandIdx = 0;
      })
      .template Case<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcFhwcOp,
                     linalg::Conv1DNcwFcwOp>([&](auto op) {
        weights = op.getOperand(1);
        weightOperandIdx = 1;
        matchDim = 0;
      })
      .template Case<linalg::Conv2DNhwcHwcfOp>([&](auto op) {
        weights = op.getOperand(1);
        weightOperandIdx = 1;
        matchDim = 3;
      })
      .template Case<linalg::Conv1DNwcWcfOp>([&](auto op) {
        weights = op.getOperand(1);
        weightOperandIdx = 1;
        matchDim = 2;
      })
      .Default([](auto) {});

  if (!weights) return failure();

  auto weightsType = cast<RankedTensorType>(weights.getType());
  auto scaleValType = cast<RankedTensorType>(scale_val.getType());

  if (scaleValType.getRank() != 1) return failure();

  // Determine which dimension of the weight matrix we need to broadcast the
  // scalar along.
  if (matchDim == -1) {
    for (int i = 0; i < weightsType.getRank(); ++i) {
      if (weightsType.getDimSize(i) == scaleValType.getDimSize(0)) {
        matchDim = i;
        break;
      }
    }
  }
  if (matchDim == -1) return failure();

  SmallVector<int64_t> addedDims;
  for (int i = 0; i < weightsType.getRank(); ++i) {
    if (i != matchDim) {
      addedDims.push_back(i);
    }
  }

  auto emptyOp = tensor::EmptyOp::create(rewriter, linearOp->getLoc(),
                                         weightsType.getShape(),
                                         weightsType.getElementType());

  auto broadcastOp = linalg::BroadcastOp::create(
      rewriter, linearOp->getLoc(), scale_val, emptyOp.getResult(), addedDims);

  auto scaledWeights =
      OpTy::create(rewriter, op.getLoc(), weights, broadcastOp.getResults()[0]);

  IRMapping bvm;
  Operation* newLinearOp = rewriter.clone(*linearOp, bvm);
  newLinearOp->setOperand(weightOperandIdx, scaledWeights);

  rewriter.replaceOp(op, newLinearOp->getResults());
  return success();
}

template <typename OpTy>
LogicalResult fuseAddOrSubIntoLinearOp(PatternRewriter& rewriter, OpTy op) {
  Operation* linearOp = nullptr;
  Value addend;
  if (failed(findLinearOpAndOperand(op, linearOp, addend))) return failure();

  auto destStyleOp = dyn_cast<DestinationStyleOpInterface>(linearOp);
  if (!destStyleOp) return failure();

  if (destStyleOp.getNumDpsInits() != 1) return failure();

  auto outputType =
      cast<RankedTensorType>(destStyleOp.getDpsInitOperand(0)->get().getType());
  auto addendType = cast<RankedTensorType>(addend.getType());

  if (addendType.getRank() != 1) return failure();

  int64_t matchDimAddend = -1;
  for (int i = 0; i < outputType.getRank(); ++i) {
    if (outputType.getDimSize(i) == addendType.getDimSize(0)) {
      matchDimAddend = i;
      break;
    }
  }
  if (matchDimAddend == -1) return failure();

  SmallVector<int64_t> addedDimsAddend;
  for (int i = 0; i < outputType.getRank(); ++i) {
    if (i != matchDimAddend) {
      addedDimsAddend.push_back(i);
    }
  }

  auto emptyOutputOp = tensor::EmptyOp::create(rewriter, linearOp->getLoc(),
                                               outputType.getShape(),
                                               outputType.getElementType());

  auto broadcastAddendOp =
      linalg::BroadcastOp::create(rewriter, linearOp->getLoc(), addend,
                                  emptyOutputOp.getResult(), addedDimsAddend);

  Value existingOuts = destStyleOp.getDpsInitOperand(0)->get();
  Value newOuts;
  if (existingOuts.getDefiningOp<tensor::EmptyOp>()) {
    if constexpr (std::is_same_v<OpTy, arith::SubFOp>) {
      newOuts = arith::NegFOp::create(rewriter, op.getLoc(),
                                      broadcastAddendOp.getResults()[0]);
    } else {
      newOuts = broadcastAddendOp.getResults()[0];
    }
  } else {
    newOuts = OpTy::create(rewriter, op.getLoc(), existingOuts,
                           broadcastAddendOp.getResults()[0]);
  }

  IRMapping bvm;
  Operation* newLinearOp = rewriter.clone(*linearOp, bvm);
  unsigned initOperandIdx =
      destStyleOp.getDpsInitOperand(0)->getOperandNumber();
  newLinearOp->setOperand(initOperandIdx, newOuts);

  rewriter.replaceOp(op, newLinearOp->getResults());
  return success();
}

struct FuseScaleIntoLinearOp : public OpRewritePattern<arith::MulFOp> {
  using OpRewritePattern<arith::MulFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulFOp op,
                                PatternRewriter& rewriter) const override {
    return fuseScaleOrDivIntoLinearOp(rewriter, op);
  }
};

struct FuseDivIntoLinearOp : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern<arith::DivFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter& rewriter) const override {
    return fuseScaleOrDivIntoLinearOp(rewriter, op);
  }
};

struct FuseAddIntoLinearOp : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern<arith::AddFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter& rewriter) const override {
    return fuseAddOrSubIntoLinearOp(rewriter, op);
  }
};

struct FuseSubIntoLinearOp : public OpRewritePattern<arith::SubFOp> {
  using OpRewritePattern<arith::SubFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::SubFOp op,
                                PatternRewriter& rewriter) const override {
    return fuseAddOrSubIntoLinearOp(rewriter, op);
  }
};

}  // namespace

struct LinalgFuseLinearOps
    : public impl::LinalgFuseLinearOpsBase<LinalgFuseLinearOps> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FuseScaleIntoLinearOp, FuseDivIntoLinearOp,
                 FuseAddIntoLinearOp, FuseSubIntoLinearOp>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createLinalgFuseLinearOpsPass() {
  return std::make_unique<LinalgFuseLinearOps>();
}

}  // namespace heir
}  // namespace mlir
