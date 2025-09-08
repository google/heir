#include "lib/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "tensor-ext-to-tensor"

namespace mlir::heir::tensor_ext {

#define GEN_PASS_DEF_TENSOREXTTOTENSOR
#include "lib/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

// Implement tensor_ext.rotate in terms of tensor.extract_slice and
// tensor.insert_slice.
struct ConvertRotateOp : public OpRewritePattern<RotateOp> {
  ConvertRotateOp(mlir::MLIRContext *context)
      : OpRewritePattern<RotateOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RotateOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType tensorType =
        dyn_cast<RankedTensorType>(op.getTensor().getType());
    if (!tensorType) {
      return rewriter.notifyMatchFailure(op, "only ranked tensors supported");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // For a fully general implementation, in particular if the shift is an SSA
    // value we have no static information about, we need to ensure the shift
    // is normalized in the range of the tensor's last dim.
    int64_t lastDimSize = tensorType.getDimSize(tensorType.getRank() - 1);
    OpFoldResult shift = getAsOpFoldResult(op.getShift());
    Type shiftType =
        (isa<Value>(shift) ? dyn_cast<Value>(shift).getType()
                           : cast<TypedAttr>(cast<Attribute>(shift)).getType());
    OpFoldResult normalizedShift;
    Value dimSize = arith::ConstantOp::create(
        b, rewriter.getIntegerAttr(shiftType, lastDimSize));

    if (auto shiftAttr = dyn_cast<Attribute>(shift)) {
      int64_t shift = cast<IntegerAttr>(shiftAttr).getInt();
      int64_t normalized = shift % lastDimSize;
      if (normalized < 0) {
        normalized += lastDimSize;
      }
      normalizedShift = rewriter.getIntegerAttr(shiftType, normalized);
    } else {
      // normalizedShift = (shift % lastDimSize + lastDimSize) % lastDimSize
      auto mod = arith::RemSIOp::create(b, op.getShift(), dimSize);
      auto add = arith::AddIOp::create(b, mod, dimSize);
      Value normalizedIntShift =
          arith::RemSIOp::create(b, add, dimSize).getResult();
      // ExtractSliceOp requires an index type, so cast it to an index
      // if needed.
      if (!normalizedIntShift.getType().isIndex()) {
        normalizedIntShift = arith::IndexCastOp::create(
            b, rewriter.getIndexType(), normalizedIntShift);
      }
      normalizedShift = normalizedIntShift;
    }

    // Now a rotation is elementwise on all but the last dimension, so we can
    // apply a slice extraction where only the last dimension is sliced up to
    // the rotation point, and a second extraction from the rotation point to
    // the end, then insert them in swapped order.

    // Offsets are all zeros except for the rightSlice's last dim, which starts
    // at the first index shifted left that does not wrap around.
    SmallVector<OpFoldResult> leftSliceOffsets;
    SmallVector<OpFoldResult> rightSliceOffsets;
    for (int i = 0; i < tensorType.getRank() - 1; ++i) {
      leftSliceOffsets.push_back(b.getIndexAttr(0));
      rightSliceOffsets.push_back(b.getIndexAttr(0));
    }
    leftSliceOffsets.push_back(b.getIndexAttr(0));
    rightSliceOffsets.push_back(normalizedShift);

    // Sizes are all the full dim size, except for the last dim, where the left
    // slice goes up to the rotation amount and the right slice goes from there
    // to the end.
    SmallVector<OpFoldResult> leftSliceSizes;
    SmallVector<OpFoldResult> rightSliceSizes;
    for (int i = 0; i < tensorType.getRank() - 1; ++i) {
      leftSliceSizes.push_back(b.getIndexAttr(tensorType.getDimSize(i)));
      rightSliceSizes.push_back(b.getIndexAttr(tensorType.getDimSize(i)));
    }
    leftSliceSizes.push_back(normalizedShift);

    OpFoldResult dimMinusShift;
    if (auto nsAttr = dyn_cast<Attribute>(normalizedShift)) {
      int64_t ns = cast<IntegerAttr>(nsAttr).getInt();
      dimMinusShift = b.getIndexAttr(lastDimSize - ns);
    } else {
      Value dimSizeCast =
          (dimSize.getType().isIndex()
               ? dimSize
               : arith::IndexCastOp::create(b, rewriter.getIndexType(), dimSize)
                     .getResult());
      dimMinusShift =
          arith::SubIOp::create(b, dimSizeCast, cast<Value>(normalizedShift))
              .getResult();
    }
    rightSliceSizes.push_back(dimMinusShift);

    SmallVector<OpFoldResult> allOneStrides(tensorType.getRank(),
                                            b.getIndexAttr(1));

    LLVM_DEBUG({
      llvm::dbgs() << "leftSliceOffsets:\n";
      for (auto ofr : leftSliceOffsets) {
        ofr.dump();
      }
      llvm::dbgs() << "rightSliceOffsets:\n";
      for (auto ofr : rightSliceOffsets) {
        ofr.dump();
      }
      llvm::dbgs() << "leftSliceSizes:\n";
      for (auto ofr : leftSliceSizes) {
        ofr.dump();
      }
      llvm::dbgs() << "rightSliceSizes:\n";
      for (auto ofr : rightSliceSizes) {
        ofr.dump();
      }
    });

    auto left = tensor::ExtractSliceOp::create(
        b, op.getTensor(), leftSliceOffsets, leftSliceSizes, allOneStrides);
    auto right = tensor::ExtractSliceOp::create(
        b, op.getTensor(), rightSliceOffsets, rightSliceSizes, allOneStrides);

    // For the insertion, the left slice goes at the back (starting at dim -
    // shift) and the right slice goes at the front.
    leftSliceOffsets[leftSliceOffsets.size() - 1] = dimMinusShift;
    rightSliceOffsets[rightSliceOffsets.size() - 1] = b.getIndexAttr(0);

    auto empty =
        tensor::EmptyOp::create(rewriter, op.getLoc(), tensorType.getShape(),
                                tensorType.getElementType());
    auto insertedLeftSlice = tensor::InsertSliceOp::create(
        b, left.getResult(), empty, leftSliceOffsets, leftSliceSizes,
        allOneStrides);
    auto insertedRightSlice = tensor::InsertSliceOp::create(
        b, right.getResult(), insertedLeftSlice.getResult(), rightSliceOffsets,
        rightSliceSizes, allOneStrides);

    rewriter.replaceOp(op, insertedRightSlice);
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
