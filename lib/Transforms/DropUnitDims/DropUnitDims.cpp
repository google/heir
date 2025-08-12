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
/// tensor of rank `rank` at positions in `positions`.
static SmallVector<ReassociationIndices> getReassociationForReshapeAtDim(
    int64_t rank, ArrayRef<int64_t> positions) {
  SmallVector<ReassociationIndices> reassociation;
  reassociation.reserve(rank - positions.size());

  llvm::DenseMap<int64_t, bool> positionsMap;
  for (int64_t pos : positions) {
    positionsMap[pos] = true;
  }
  auto isUnitDim = [&](int64_t dim) { return positionsMap.contains(dim); };

  ReassociationIndices reassociationGroup;
  unsigned dim = 0;
  while (dim < rank && isUnitDim(dim)) reassociationGroup.push_back(dim++);
  while (dim < rank) {
    assert(!isUnitDim(dim) && "expected non unit-extent");
    reassociationGroup.push_back(dim);
    ++dim;
    // Fold all following dimensions that are unit-extent.
    while (dim < rank && isUnitDim(dim)) {
      reassociationGroup.push_back(dim++);
    }
    reassociation.push_back(reassociationGroup);
    reassociationGroup.clear();
  }
  return reassociation;
}

/// Collapse the given `value` so that the type matches the type of
/// `origOutput`.
static Value collapseValue(RewriterBase& rewriter, Location loc, Value operand,
                           ArrayRef<int64_t> targetShape,
                           ArrayRef<ReassociationIndices> reassociation) {
  auto tensorType = cast<RankedTensorType>(operand.getType());
  auto targetType =
      RankedTensorType::get(targetShape, tensorType.getElementType());
  return tensor::CollapseShapeOp::create(rewriter, loc, targetType, operand,
                                         reassociation);
}

/// Returns a collapsed `val` where the collapsing occurs at dims in positions.
static Value collapseDimsAt(PatternRewriter& rewriter, Value val,
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

}  // namespace

// Drop unit dims on linalg.map operations that perform a single elementwise
// operation. This will only drop batch dims (leading unit dimensions). This
// pass is inspired by the anonymous base class RankReduceContractionOps in
// llvm-project/mlir/lib/Dialect/Linalg/Transforms/DropUnitDims.cpp, but is
// heavily simplified.
struct ReduceLinalgMap : OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  /// Collapse all collapsible operands.
  SmallVector<Value> collapseOperands(PatternRewriter& rewriter,
                                      ArrayRef<Value> operands,
                                      ArrayRef<int64_t> collapseDims) const {
    return llvm::map_to_vector(operands, [&](auto operand) {
      return collapseDimsAt(rewriter, operand, collapseDims);
    });
  }

  /// Expand result tensor.
  Value expandResult(PatternRewriter& rewriter, Value result,
                     RankedTensorType expandedType,
                     SmallVector<int64_t> dims) const {
    return tensor::ExpandShapeOp::create(
        rewriter, result.getLoc(), expandedType, result,
        getReassociationForReshapeAtDim(expandedType.getRank(), dims));
  }

  LogicalResult matchAndRewrite(linalg::MapOp mapOp,
                                PatternRewriter& rewriter) const override {
    if (mapOp.hasUserDefinedMaps()) {
      return rewriter.notifyMatchFailure(
          mapOp, "ops with user-defined maps are not supported");
    }

    // The mapper should have exactly two operations (the second is a yield).
    auto mapper = mapOp.getBody();
    if (mapper->getOperations().size() != 2) return failure();

    // The operation should be elementwise.
    Operation& mappingOp = mapper->getOperations().front();
    if (!mappingOp.hasTrait<mlir::OpTrait::Elementwise>()) {
      return failure();
    }

    auto loc = mapOp.getLoc();
    auto inputs = mapOp.getDpsInputs();
    SmallVector<Value> operands = inputs;
    operands.push_back(mapOp.getInit());

    // Check for unit dims in the output shape. A map op requires all inputs and
    // outputs have the same shape.
    auto outputShape = mapOp.getInit().getType().getShape();
    SmallVector<int64_t> operandUnitDims;
    for (int64_t i = 0; i < outputShape.size(); ++i) {
      if (outputShape[i] == 1) {
        operandUnitDims.push_back(i);
      }
    }

    if (operandUnitDims.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "no unit dims to drop");
      return failure();
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

struct DropUnitDims : impl::DropUnitDimsBase<DropUnitDims> {
  using DropUnitDimsBase::DropUnitDimsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateContractionOpRankReducingPatterns(patterns);
    patterns.add<ReduceLinalgMap>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
