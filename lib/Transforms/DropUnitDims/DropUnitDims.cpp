#include "lib/Transforms/DropUnitDims/DropUnitDims.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "drop-unit-dims"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_DROPUNITDIMS
#include "lib/Transforms/DropUnitDims/DropUnitDims.h.inc"

namespace {

Value collapseValue(RewriterBase& rewriter, Location loc, Value operand,
                    ArrayRef<int64_t> targetShape,
                    ArrayRef<ReassociationIndices> reassociation) {
  auto tensorType = cast<RankedTensorType>(operand.getType());
  auto targetType =
      RankedTensorType::get(targetShape, tensorType.getElementType());
  return tensor::CollapseShapeOp::create(rewriter, loc, targetType, operand,
                                         reassociation);
}

}  // namespace

SmallVector<int64_t> getUnitDims(ShapedType type) {
  SmallVector<int64_t> unitDims;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.getDimSize(i) == 1) {
      unitDims.push_back(i);
    }
  }
  return unitDims;
}

Value collapseDimsAt(PatternRewriter& rewriter, Value val,
                     ArrayRef<int64_t> positions) {
  auto valType = cast<ShapedType>(val.getType());
  SmallVector<int64_t> collapsedShape(valType.getShape());
  for (int64_t pos : llvm::reverse(positions)) {
    collapsedShape.erase(collapsedShape.begin() + pos);
  }
  return collapseValue(
      rewriter, val.getLoc(), val, collapsedShape,
      getReassociationForReshapeAtDim(valType.getRank(), positions));
}

/// Collapse all collapsible operands.
SmallVector<Value> collapseOperands(PatternRewriter& rewriter,
                                    ArrayRef<Value> operands,
                                    ArrayRef<int64_t> collapseDims) {
  return llvm::map_to_vector(operands, [&](auto operand) {
    return collapseDimsAt(rewriter, operand, collapseDims);
  });
}

/// Expand result tensor.
Value expandResult(PatternRewriter& rewriter, Value result,
                   RankedTensorType expandedType, SmallVector<int64_t> dims) {
  return tensor::ExpandShapeOp::create(
      rewriter, result.getLoc(), expandedType, result,
      getReassociationForReshapeAtDim(expandedType.getRank(), dims));
}

// Drop unit dims on linalg.map operations that perform a single elementwise
// operation. This will only drop batch dims (leading unit dimensions). This
// pass is inspired by the anonymous base class RankReduceContractionOps in
// llvm-project/mlir/lib/Dialect/Linalg/Transforms/DropUnitDims.cpp, but is
// heavily simplified.
struct ReduceLinalgMap : OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MapOp mapOp,
                                PatternRewriter& rewriter) const override {
    if (mapOp.hasUserDefinedMaps()) {
      return rewriter.notifyMatchFailure(
          mapOp, "ops with user-defined maps are not supported");
    }

    // The mapper should have exactly two operations (the second is a yield).
    auto mapper = mapOp.getBody();
    if (mapper->getOperations().size() != 2)
      return rewriter.notifyMatchFailure(
          mapOp, "mapper block must have exactly two operations");

    // The operation should be elementwise.
    Operation& mappingOp = mapper->getOperations().front();
    if (!mappingOp.hasTrait<mlir::OpTrait::Elementwise>()) {
      return rewriter.notifyMatchFailure(
          mapOp, "mapping op must have elementwise trait");
    }

    auto loc = mapOp.getLoc();
    auto inputs = mapOp.getDpsInputs();
    SmallVector<Value> operands = inputs;
    operands.push_back(mapOp.getInit());

    // Check for unit dims in the output shape. A map op requires all inputs and
    // outputs have the same shape.
    SmallVector<int64_t> operandUnitDims =
        getUnitDims(mapOp.getInit().getType());
    if (operandUnitDims.empty()) {
      return rewriter.notifyMatchFailure(mapOp, "no unit dims to drop");
    }

    LLVM_DEBUG({
      llvm::dbgs() << "found unit dims: ";
      for (auto dim : operandUnitDims) {
        llvm::dbgs() << dim << ", ";
      }
      llvm::dbgs() << "\n";
    });

    SmallVector<Value> collapsedOperands =
        collapseOperands(rewriter, operands, operandUnitDims);

    Value collapsedInit = collapsedOperands.back();
    SmallVector<Value> collapsedInputs = {collapsedOperands.begin(),
                                          collapsedOperands.end() - 1};
    SmallVector<Type, 1> collapsedResultTy;
    collapsedResultTy.push_back(collapsedInit.getType());
    linalg::MapOp collapsedOp = linalg::MapOp::create(
        rewriter, loc, collapsedInputs, collapsedInit,
        [&](OpBuilder& b, Location loc, ValueRange blockArguments) {
          IRMapping mp;
          for (BlockArgument blockArg : mapper->getArguments()) {
            mp.map(blockArg, blockArguments[blockArg.getArgNumber()]);
          }
          for (auto& op : mapper->getOperations()) {
            b.clone(op, mp);
          }
        },
        mapOp->getAttrs());

    rewriter.replaceOp(
        mapOp, expandResult(rewriter, collapsedOp.getResult()[0],
                            cast<RankedTensorType>(mapOp.getInit().getType()),
                            operandUnitDims));
    return success();
  }
};

struct CollapseExtractSlice : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto sourceType = op.getSourceType();
    auto resultType = op.getResultType();
    SliceVerificationResult res = isRankReducedType(sourceType, resultType);
    if (res != SliceVerificationResult::Success)
      return rewriter.notifyMatchFailure(
          op, "expected rank reducing extract_slice");

    auto reassociation =
        getReassociationIndicesForReshape(sourceType, resultType);
    if (!reassociation)
      return rewriter.notifyMatchFailure(op,
                                         "failed to get reassociation indices");

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        op, op.getResultType(), op.getSource(), reassociation.value());
    return success();
  }
};

struct ExpandInsertSlice : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto sourceType = op.getSourceType();
    auto resultType = op.getResultType();
    SliceVerificationResult res = isRankReducedType(resultType, sourceType);
    if (res != SliceVerificationResult::Success)
      return rewriter.notifyMatchFailure(
          op, "expected rank expanding insert_slice");

    auto reassociation =
        getReassociationIndicesForReshape(resultType, sourceType);
    if (!reassociation)
      return rewriter.notifyMatchFailure(op,
                                         "failed to get reassociation indices");

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultType(), op.getSource(), reassociation.value());
    return success();
  }
};

struct DropUnitDims : impl::DropUnitDimsBase<DropUnitDims> {
  using DropUnitDimsBase::DropUnitDimsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateContractionOpRankReducingPatterns(patterns);
    patterns.add<ReduceLinalgMap>(context);
    linalg::ControlDropUnitDims options;
    options.rankReductionStrategy =
        linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    linalg::populateMoveInitOperandsToInputPattern(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();

    // Convert extract_slice and insert_slice to collapse and expand shape ops.
    RewritePatternSet collapsePatterns(context);
    collapsePatterns.add<CollapseExtractSlice, ExpandInsertSlice>(context);
    walkAndApplyPatterns(getOperation(), std::move(collapsePatterns));
  }
};

}  // namespace heir
}  // namespace mlir
