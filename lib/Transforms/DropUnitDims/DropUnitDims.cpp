#include "lib/Transforms/DropUnitDims/DropUnitDims.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "drop-unit-dims"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_DROPUNITDIMS
#include "lib/Transforms/DropUnitDims/DropUnitDims.h.inc"

namespace {

/// The following functions are copied from
/// llvm-project/mlir/lib/Dialect/Linalg/Transforms/DropUnitDims.cpp, where they
/// are in an anonymous namespace.

/// Returns reassociation indices for collapsing/expanding a
/// tensor of rank `rank` at position `pos`.
static SmallVector<ReassociationIndices> getReassociationForReshapeAtDim(
    int64_t rank, int64_t pos) {
  SmallVector<ReassociationIndices> reassociation(rank - 1, {0, 1});
  bool lastDim = pos == rank - 1;
  if (rank > 2) {
    for (int64_t i = 0; i < rank - 1; i++) {
      if (i == pos || (lastDim && i == pos - 1))
        reassociation[i] = ReassociationIndices{i, i + 1};
      else if (i < pos)
        reassociation[i] = ReassociationIndices{i};
      else
        reassociation[i] = ReassociationIndices{i + 1};
    }
  }
  return reassociation;
}

/// Collapse the given `value` so that the type matches the type of
/// `origOutput`.
static Value collapseValue(RewriterBase &rewriter, Location loc, Value operand,
                           ArrayRef<int64_t> targetShape,
                           ArrayRef<ReassociationIndices> reassociation) {
  auto tensorType = cast<RankedTensorType>(operand.getType());
  auto targetType =
      RankedTensorType::get(targetShape, tensorType.getElementType());
  return rewriter.create<tensor::CollapseShapeOp>(loc, targetType, operand,
                                                  reassociation);
}

/// Returns a collapsed `val` where the collapsing occurs at dim `pos`.
/// If `pos < 0`, then don't collapse.
static Value collapseSingletonDimAt(PatternRewriter &rewriter, Value val,
                                    int64_t pos) {
  if (pos < 0) return val;
  auto valType = cast<ShapedType>(val.getType());
  SmallVector<int64_t> collapsedShape(valType.getShape());
  collapsedShape.erase(collapsedShape.begin() + pos);
  return collapseValue(rewriter, val.getLoc(), val, collapsedShape,
                       getReassociationForReshapeAtDim(valType.getRank(), pos));
}

}  // namespace

// Drop unit dims on linalg.map operations that perform a single elementwise
// operation. This will only drop batch dims (leading unit dimensions). This
// pass is inspired by the anonymous base class RankReduceContractionOps in
// llvm-project/mlir/lib/Dialect/Linalg/Transforms/DropUnitDims.cpp, but is
// heavily simplified.
struct ReduceLinalgMap : OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  /// Collapse all collapsible operands.
  SmallVector<Value> collapseOperands(
      PatternRewriter &rewriter, ArrayRef<Value> operands,
      ArrayRef<int64_t> operandCollapseDims) const {
    assert(operandCollapseDims.size() == operands.size() &&
           "expected equal operands and dims sizes");
    return llvm::map_to_vector(
        llvm::zip(operands, operandCollapseDims), [&](auto pair) {
          return collapseSingletonDimAt(rewriter, std::get<0>(pair),
                                        std::get<1>(pair));
        });
  }

  /// Expand result tensor.
  Value expandResult(PatternRewriter &rewriter, Value result,
                     RankedTensorType expandedType, int64_t dim) const {
    return rewriter.create<tensor::ExpandShapeOp>(
        result.getLoc(), expandedType, result,
        getReassociationForReshapeAtDim(expandedType.getRank(), dim));
  }

  LogicalResult matchAndRewrite(linalg::MapOp mapOp,
                                PatternRewriter &rewriter) const override {
    if (mapOp.hasUserDefinedMaps()) {
      return rewriter.notifyMatchFailure(
          mapOp, "ops with user-defined maps are not supported");
    }

    // The mapper should have exactly two operations (the second is a yield).
    auto mapper = mapOp.getBody(0);
    if (mapper->getOperations().size() != 2) return failure();

    // The operation should be elementwise.
    Operation &mappingOp = mapper->getOperations().front();
    if (!mappingOp.hasTrait<mlir::OpTrait::Elementwise>()) {
      return failure();
    }

    auto loc = mapOp.getLoc();
    auto inputs = mapOp.getDpsInputs();
    auto inits = mapOp.getDpsInits();
    if (inits.size() != 1)
      return rewriter.notifyMatchFailure(mapOp, "expected 1 init");
    SmallVector<Value> operands = inputs;
    operands.push_back(inits[0]);

    // Check that the initial dim is a unit dims for all operands
    SmallVector<std::pair<Value, unsigned>> bOperands;
    mapOp.mapIterationSpaceDimToAllOperandDims(/*dimPos=*/0, bOperands);
    if (llvm::any_of(bOperands, [](auto pair) {
          return cast<ShapedType>(std::get<0>(pair).getType())
                     .getShape()[std::get<1>(pair)] != 1;
        })) {
      LLVM_DEBUG(llvm::dbgs() << "specified unit dims not found");
      return failure();
    }

    SmallVector<int64_t> operandUnitDims = llvm::to_vector(llvm::map_to_vector(
        bOperands, [](auto pair) -> int64_t { return std::get<1>(pair); }));
    SmallVector<Value> collapsedOperands =
        collapseOperands(rewriter, operands, operandUnitDims);

    Value collapsedInit = collapsedOperands.back();
    SmallVector<Value> collapsedInputs = {collapsedOperands.begin(),
                                          collapsedOperands.end() - 1};
    SmallVector<Type, 1> collapsedResultTy;
    collapsedResultTy.push_back(collapsedInit.getType());
    linalg::MapOp collapsedOp = rewriter.create<linalg::MapOp>(
        loc, collapsedInputs, collapsedInit,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value innerResult =
              rewriter
                  .create(loc, mappingOp.getName().getIdentifier(), args,
                          {mappingOp.getResultTypes()[0]}, mappingOp.getAttrs(),
                          {}, {})
                  ->getResults()[0];
          b.create<linalg::YieldOp>(loc, innerResult);
        },
        mapOp->getAttrs());

    rewriter.replaceOp(
        mapOp,
        expandResult(rewriter, collapsedOp.getResult()[0],
                     cast<RankedTensorType>(mapOp.getResult()[0].getType()),
                     operandUnitDims[0]));
    return success();
  }
};

struct DropUnitDims : impl::DropUnitDimsBase<DropUnitDims> {
  using DropUnitDimsBase::DropUnitDimsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateContractionOpRankReducingPatterns(patterns);
    patterns.add<ReduceLinalgMap>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
