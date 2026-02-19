#include "lib/Transforms/FoldConstantTensors/FoldConstantTensors.h"

#include <cstdint>
#include <utility>

#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "fold-constant-tensors"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FOLDCONSTANTTENSORS
#include "lib/Transforms/FoldConstantTensors/FoldConstantTensors.h.inc"

/// Pattern to fold an insert op of a constant destination and scalar to a new
/// constant.
///
/// Example:
/// ```
///   %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
///   %c0 = arith.constant 0 : index
///   %c4_f32 = arith.constant 4.0 : f32
///   %1 = tensor.insert %c4_f32 into %0[%c0] : tensor<4xf32>
/// ```
/// is rewritten into:
/// ```
///   %1 = arith.constant dense<[4.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
/// ```
class InsertAfterConstant final : public OpRewritePattern<tensor::InsertOp> {
 public:
  using OpRewritePattern<tensor::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertOp insertOp,
                                PatternRewriter& rewriter) const override {
    // Requires a ranked tensor type.
    auto destType =
        llvm::dyn_cast<RankedTensorType>(insertOp.getDest().getType());
    if (!destType)
      return rewriter.notifyMatchFailure(insertOp,
                                         "destination must be a ranked tensor");

    // Pattern requires constant indices
    SmallVector<uint64_t> indices;
    for (OpFoldResult indice : getAsOpFoldResult(insertOp.getIndices())) {
      auto indiceAttr = dyn_cast<Attribute>(indice);
      if (!indiceAttr)
        return rewriter.notifyMatchFailure(insertOp,
                                           "indices must be constant");
      indices.push_back(llvm::cast<IntegerAttr>(indiceAttr).getInt());
    }

    // Requires a constant scalar to insert
    OpFoldResult scalar = getAsOpFoldResult(insertOp.getScalar());
    Attribute scalarAttr = dyn_cast<Attribute>(scalar);
    if (!scalarAttr)
      return rewriter.notifyMatchFailure(insertOp,
                                         "scalar to insert must be a constant");

    if (auto constantOp = dyn_cast_or_null<arith::ConstantOp>(
            insertOp.getDest().getDefiningOp())) {
      if (auto sourceAttr =
              llvm::dyn_cast<ElementsAttr>(constantOp.getValue())) {
        // Update the attribute at the inserted index.
        auto sourceValues = sourceAttr.getValues<Attribute>();
        auto flattenedIndex = sourceAttr.getFlattenedIndex(indices);
        SmallVector<Attribute> updatedValues{sourceValues.begin(),
                                             sourceValues.end()};
        updatedValues[flattenedIndex] = scalarAttr;
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            insertOp, sourceAttr.getType(),
            DenseElementsAttr::get(cast<ShapedType>(sourceAttr.getType()),
                                   updatedValues));
        return success();
      }
    }

    return rewriter.notifyMatchFailure(
        insertOp, "destination of insert must be a constant");
  }
};

/// Pattern to fold an insertion ops into a destination tensor originating from
/// a from_elements op into a new from_elements op. This is a common pattern
/// that appears after lowering linalg to affine loops. An initial constant
/// tensor is created as the destination of the linalg operation, and subsequent
/// insertions write over the entire result.
///
/// Example:
/// ```
///   %cst = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
///   %1 = tensor.insert %c4_f32 into %cst[%c0] : tensor<2xf32>
///   %2 = tensor.insert %c8_f32 into %1[%c1] : tensor<2xf32>
/// ```
/// is rewritten into:
/// ```
///   %2 = tensor.from_elements %c4_f32, %c8_f32 : tensor<2xf32>
/// ```
class InsertIntoFromElements final : public OpRewritePattern<tensor::InsertOp> {
 public:
  using OpRewritePattern<tensor::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertOp insertOp,
                                PatternRewriter& rewriter) const override {
    // Note: we match on the first insertion after the constant, instead of the
    // later or last ones. This avoid matching on every insertion in the chain.
    auto dest = insertOp.getDest();
    auto destType = llvm::dyn_cast<RankedTensorType>(dest.getType());
    if (!destType)
      return rewriter.notifyMatchFailure(insertOp,
                                         "destination must be a ranked tensor");

    auto constantOp = dyn_cast_or_null<arith::ConstantOp>(dest.getDefiningOp());
    if (!constantOp)
      return rewriter.notifyMatchFailure(
          insertOp, "destination of insert must be a constant");

    // Collect the element at the index, and then accumulate all the insertion
    // uses. Each insertion use should only have one use. Once we collect all
    // elements, then replace with a from_elements op.
    DenseMap<int64_t, Value> flatIndexToElement;
    tensor::InsertOp currentInsertOp = insertOp;
    SmallVector<Operation*> opsToErase;
    while (currentInsertOp) {
      auto maybeFlatIndex = getFlattenedIndex(
          destType, getAsOpFoldResult(currentInsertOp.getIndices()));
      if (failed(maybeFlatIndex))
        return rewriter.notifyMatchFailure(currentInsertOp,
                                           "failed to compute flattened index");

      // Overriding values in the tensor is not supported.
      auto flatIndex = maybeFlatIndex.value();
      if (flatIndexToElement.contains(flatIndex)) return failure();
      flatIndexToElement[flatIndex] = currentInsertOp.getScalar();
      opsToErase.push_back(currentInsertOp);

      if (!currentInsertOp->hasOneUse())
        return rewriter.notifyMatchFailure(
            currentInsertOp, "insert op must have exactly one use");
      auto nextOp = *currentInsertOp->getUsers().begin();
      currentInsertOp = dyn_cast<tensor::InsertOp>(nextOp);
    }

    SmallVector<Value> values;
    auto constantValue = cast<DenseElementsAttr>(constantOp.getValue());
    for (auto i = 0; i < destType.getNumElements(); ++i) {
      // Add the inserted value if it exists, or the value from the original
      // constant.
      if (flatIndexToElement.contains(i)) {
        values.push_back(flatIndexToElement[i]);
      } else {
        values.push_back(arith::ConstantOp::create(
            rewriter, insertOp.getLoc(), constantValue.getElementType(),
            cast<TypedAttr>(constantValue.getValues<Attribute>()[i])));
      }
    }

    rewriter.replaceAllUsesWith(
        opsToErase.back()->getResult(0),
        tensor::FromElementsOp::create(rewriter, insertOp.getLoc(), destType,
                                       values)
            .getResult());
    for (auto op : llvm::reverse(opsToErase)) {
      op->erase();
    }
    return success();
  }
};

/// Pattern to fold an collapse op of a constant to a constant with a collapsed
/// shape.
///
/// Example:
/// ```
///   %0 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
///   %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<1x4xf32> into
///   tensor<4xf32>
/// ```
/// is rewritten into:
/// ```
///   %1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
/// ```
class CollapseShapeAfterConstant final
    : public OpRewritePattern<tensor::CollapseShapeOp> {
 public:
  using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter& rewriter) const override {
    auto constantOp = dyn_cast_or_null<arith::ConstantOp>(
        collapseOp.getSrc().getDefiningOp());
    if (!constantOp)
      return rewriter.notifyMatchFailure(
          collapseOp, "source of collapse must be a constant");

    auto sourceAttr = llvm::dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (!sourceAttr)
      return rewriter.notifyMatchFailure(
          collapseOp, "source of collapse must be an elements attribute");

    auto resultTy = collapseOp.getResult().getType();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        collapseOp, resultTy, DenseElementsAttr::get(resultTy, sourceAttr));
    return success();
  }
};

struct CollapseEmptyTensor
    : public OpRewritePattern<mlir::tensor::CollapseShapeOp> {
 public:
  CollapseEmptyTensor(MLIRContext* context)
      : OpRewritePattern<mlir::tensor::CollapseShapeOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tensor::CollapseShapeOp collapseOp,
                                PatternRewriter& rewriter) const override {
    auto emptyOp =
        dyn_cast_or_null<tensor::EmptyOp>(collapseOp.getSrc().getDefiningOp());
    if (!emptyOp)
      return rewriter.notifyMatchFailure(
          collapseOp, "source of collapse must be an empty tensor");

    auto resultTy = collapseOp.getResult().getType();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        collapseOp, resultTy.getShape(), resultTy.getElementType());
    return success();
  }
};

struct ExtractSliceOfSplat
    : public OpRewritePattern<mlir::tensor::ExtractSliceOp> {
 public:
  ExtractSliceOfSplat(MLIRContext* context)
      : OpRewritePattern<mlir::tensor::ExtractSliceOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tensor::ExtractSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto splatOp =
        dyn_cast_or_null<tensor::SplatOp>(op.getSource().getDefiningOp());
    if (!splatOp)
      return rewriter.notifyMatchFailure(
          op, "source of extract slice must be a splat op");

    auto resultTy = op.getResult().getType();
    auto newSplat = tensor::SplatOp::create(
        rewriter, op.getLoc(), splatOp.getInput(), resultTy.getShape());
    rewriter.replaceOp(op, newSplat);
    return success();
  }
};

class ExtractOfExtractSlice final : public OpRewritePattern<tensor::ExtractOp> {
 public:
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  ExtractOfExtractSlice(MLIRContext* context)
      : OpRewritePattern<tensor::ExtractOp>(context) {}

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter& rewriter) const override {
    auto extractSliceOp =
        extractOp.getTensor().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractSliceOp)
      return rewriter.notifyMatchFailure(extractOp,
                                         "tensor must be an extract_slice op");

    auto sourceType = llvm::cast<ShapedType>(extractSliceOp.getSourceType());
    auto sliceType = llvm::cast<ShapedType>(extractOp.getTensor().getType());
    if (!sliceType.hasStaticShape() || !sourceType.hasStaticShape())
      return rewriter.notifyMatchFailure(extractOp,
                                         "slice type must have a static shape");

    SmallVector<Value> sourceIndices;
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, extractOp.getLoc(), extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedStrides(), extractSliceOp.getDroppedDims(),
        getAsOpFoldResult(extractOp.getIndices()), sourceIndices);

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        extractOp, extractSliceOp.getSource(), sourceIndices);
    return success();
  }
};

struct FoldConstantTensors
    : public impl::FoldConstantTensorsBase<FoldConstantTensors> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<InsertAfterConstant, CollapseShapeAfterConstant,
                 CollapseEmptyTensor, InsertIntoFromElements,
                 ExtractSliceOfSplat, ExtractOfExtractSlice>(context);

    // Run pattern matching and conversion
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
