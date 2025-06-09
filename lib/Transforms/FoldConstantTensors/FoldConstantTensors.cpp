#include "lib/Transforms/FoldConstantTensors/FoldConstantTensors.h"

#include <cstdint>
#include <utility>

#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
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
                                PatternRewriter &rewriter) const override {
    // Requires a ranked tensor type.
    auto destType =
        llvm::dyn_cast<RankedTensorType>(insertOp.getDest().getType());
    if (!destType) return failure();

    // Pattern requires constant indices
    SmallVector<uint64_t> indices;
    for (OpFoldResult indice : getAsOpFoldResult(insertOp.getIndices())) {
      auto indiceAttr = dyn_cast<Attribute>(indice);
      if (!indiceAttr) return failure();
      indices.push_back(llvm::cast<IntegerAttr>(indiceAttr).getInt());
    }

    // Requires a constant scalar to insert
    OpFoldResult scalar = getAsOpFoldResult(insertOp.getScalar());
    Attribute scalarAttr = dyn_cast<Attribute>(scalar);
    if (!scalarAttr) return failure();

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

    return failure();
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
                                PatternRewriter &rewriter) const override {
    auto constantOp = dyn_cast_or_null<arith::ConstantOp>(
        collapseOp.getSrc().getDefiningOp());
    if (!constantOp) return failure();

    auto sourceAttr = llvm::dyn_cast<ElementsAttr>(constantOp.getValue());
    if (!sourceAttr) return failure();

    auto resultTy = collapseOp.getResult().getType();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        collapseOp, resultTy, DenseElementsAttr::get(resultTy, sourceAttr));
    return success();
  }
};

struct CollapseEmptyTensor
    : public OpRewritePattern<mlir::tensor::CollapseShapeOp> {
 public:
  CollapseEmptyTensor(MLIRContext *context)
      : OpRewritePattern<mlir::tensor::CollapseShapeOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto emptyOp =
        dyn_cast_or_null<tensor::EmptyOp>(collapseOp.getSrc().getDefiningOp());
    if (!emptyOp) return failure();

    auto resultTy = collapseOp.getResult().getType();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        collapseOp, resultTy.getShape(), resultTy.getElementType());
    return success();
  }
};

struct FoldConstantTensors
    : public impl::FoldConstantTensorsBase<FoldConstantTensors> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<InsertAfterConstant, CollapseShapeAfterConstant,
                 CollapseEmptyTensor>(context);

    // Run pattern matching and conversion
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
