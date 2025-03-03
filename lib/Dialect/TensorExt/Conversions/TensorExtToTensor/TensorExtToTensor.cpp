#include "lib/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::tensor_ext {

#define GEN_PASS_DEF_TENSOREXTTOTENSOR
#include "lib/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

struct ConvertRotateOp : public OpRewritePattern<RotateOp> {
  ConvertRotateOp(mlir::MLIRContext *context)
      : OpRewritePattern<RotateOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RotateOp op,
                                PatternRewriter &rewriter) const override {
    auto shift = op.getShift();
    auto constantOp =
        dyn_cast_or_null<arith::ConstantOp>(shift.getDefiningOp());
    if (!constantOp) {
      return failure();
    }

    auto shiftValue = cast<IntegerAttr>(constantOp.getValue()).getInt();
    auto tensorShape =
        cast<RankedTensorType>(op.getTensor().getType()).getShape();

    // only support 1D tensors
    // TODO(#924): Currently RotateOp only supports rotating a 1-D vector, or a
    // vector with only one non-unit dimension that is treated as the major
    // dimension.
    if (tensorShape.size() != 1) {
      return failure();
    }

    auto tensorSize = tensorShape[0];
    if (shiftValue < 0) {
      shiftValue += tensorSize;
    }

    auto tensorElementType = getElementTypeOrSelf(op.getTensor().getType());
    auto leftTensorType =
        RankedTensorType::get({shiftValue}, tensorElementType);
    auto rightTensorType =
        RankedTensorType::get({tensorSize - shiftValue}, tensorElementType);

    auto left = rewriter.create<tensor::ExtractSliceOp>(
        op.getLoc(), leftTensorType, op.getTensor(), ArrayRef<Value>{},
        ArrayRef<Value>{}, ArrayRef<Value>{},
        /*offsets=*/ArrayRef<int64_t>{0},
        /*sizes=*/ArrayRef{shiftValue}, /*strides=*/ArrayRef<int64_t>{1});
    auto right = rewriter.create<tensor::ExtractSliceOp>(
        op.getLoc(), rightTensorType, op.getTensor(), ArrayRef<Value>{},
        ArrayRef<Value>{}, ArrayRef<Value>{},
        /*offsets=*/ArrayRef{shiftValue},
        /*sizes=*/ArrayRef{tensorSize - shiftValue},
        /*strides=*/ArrayRef<int64_t>{1});
    // for tensor.concat to lower we need to use
    // transform.apply_patterns.tensor.decompose_concat which is quite painful
    // auto concat = rewriter.create<tensor::ConcatOp>(
    //    op.getLoc(), /*dim=*/0,
    //    ValueRange{right.getResult(), left.getResult()});
    auto empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), tensorShape,
                                                  tensorElementType);
    auto insertLeftToRight = rewriter.create<tensor::InsertSliceOp>(
        op.getLoc(), left.getResult(), empty, ArrayRef<Value>{},
        ArrayRef<Value>{}, ArrayRef<Value>{},
        /*offsets=*/ArrayRef<int64_t>{tensorSize - shiftValue},
        /*sizes=*/ArrayRef{shiftValue}, /*strides=*/ArrayRef<int64_t>{1});
    auto insertRightToLeft = rewriter.create<tensor::InsertSliceOp>(
        op.getLoc(), right.getResult(), insertLeftToRight, ArrayRef<Value>{},
        ArrayRef<Value>{}, ArrayRef<Value>{},
        /*offsets=*/ArrayRef<int64_t>{0},
        /*sizes=*/ArrayRef{tensorSize - shiftValue},
        /*strides=*/ArrayRef<int64_t>{1});

    rewriter.replaceAllOpUsesWith(op, insertRightToLeft.getResult());
    return success();
  }
};

struct TensorExtToTensor
    : public impl::TensorExtToTensorBase<TensorExtToTensor> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ConvertRotateOp>(context);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace mlir::heir::tensor_ext
