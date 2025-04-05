#include "lib/Utils/RewriteUtils/RewriteUtils.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_TENSORLINALGTOAFFINELOOPS
#include "lib/Transforms/TensorLinalgToAffineLoops/TensorLinalgToAffineLoops.h.inc"

using affine::AffineForOp;
using affine::AffineYieldOp;
using linalg::GenericOp;
using linalg::LinalgOp;
using tensor::ExtractOp;
using tensor::InsertOp;

static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> vals) {
  if (map.isEmpty()) return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
    SmallVector<Value> operands(vals);
    affine::canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(b.create<affine::AffineApplyOp>(loc, exprMap, operands));
  }
  return res;
}

static SmallVector<Value> inlineRegionAndEmitStore(
    OpBuilder &b, Location loc, LinalgOp op, ArrayRef<Value> indexedValues,
    ArrayRef<SmallVector<Value>> indexing, ValueRange outputBuffers) {
  auto &block = op->getRegion(0).front();
  IRMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    auto *newOp = b.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  Operation *terminator = block.getTerminator();
  SmallVector<Value> insertResults;
  for (OpOperand &operand : terminator->getOpOperands()) {
    Value toStore = map.lookupOrDefault(operand.get());
    InsertOp insertion = b.create<InsertOp>(
        loc, toStore, outputBuffers[operand.getOperandNumber()],
        indexing[operand.getOperandNumber()]);
    insertResults.push_back(insertion.getResult());
  }

  return insertResults;
}

static SmallVector<Value> emitScalarImplementation(OpBuilder &b, Location loc,
                                                   ArrayRef<Value> allIvs,
                                                   LinalgOp linalgOp,
                                                   ValueRange iterArgs) {
  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp->getNumOperands());
  auto allIvsPlusDims = SmallVector<Value>(allIvs);

  // 1.a. Emit load from input operand or for scalars access the operand itself.
  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(inputOperand->get());
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(inputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<ExtractOp>(loc, inputOperand->get(), indexing));
  }
  // 1.b. Emit load from output views.
  //
  // Differing from upstream, the output view is now a region iterArg, so we
  // thread through the iterArgs of the containing loop nest, which correspond
  // to the iter args of the output tensors of the original linalgOp.
  int iterArgIndex = 0;
  for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
    SmallVector<Value> indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
        allIvsPlusDims);
    indexedValues.push_back(
        b.create<ExtractOp>(loc, iterArgs[iterArgIndex++], indexing));
  }

  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  //
  // Differing from upstream, the outputBuffer must be a region iterArg,
  // similar to 1.b.
  SmallVector<SmallVector<Value>, 8> indexing;
  for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
        allIvsPlusDims));
  }
  return inlineRegionAndEmitStore(b, loc, linalgOp, indexedValues, indexing,
                                  iterArgs);
}

static AffineForOp buildAffineLoopFromConstants(
    OpBuilder &builder, Location loc, int64_t lb, int64_t ub, int64_t step,
    ValueRange outputInits, AffineForOp::BodyBuilderFn bodyBuilderFn) {
  return builder.create<AffineForOp>(loc, lb, ub, step,
                                     /*iterArgs=*/outputInits, bodyBuilderFn);
}

static AffineForOp buildAffineLoopFromValues(
    OpBuilder &builder, Location loc, Value lb, Value ub, int64_t step,
    ValueRange outputInits, AffineForOp::BodyBuilderFn bodyBuilderFn) {
  std::optional<int64_t> lbConst = getConstantIntValue(lb);
  std::optional<int64_t> ubConst = getConstantIntValue(ub);
  if (lbConst && ubConst)
    return buildAffineLoopFromConstants(builder, loc, lbConst.value(),
                                        ubConst.value(), step, outputInits,
                                        bodyBuilderFn);
  return builder.create<AffineForOp>(loc, lb, builder.getDimIdentityMap(), ub,
                                     builder.getDimIdentityMap(), step,
                                     /*iterArgs=*/outputInits, bodyBuilderFn);
}

AffineForOp buildAffineLoopNestWithCarriedIterArgs(
    OpBuilder &builder, Location loc, ArrayRef<Value> lbs, ArrayRef<Value> ubs,
    ArrayRef<int64_t> steps, ValueRange outputInits,
    function_ref<SmallVector<Value>(OpBuilder &, Location, ValueRange,
                                    ValueRange)>
        bodyBuilderFn) {
  assert(lbs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(lbs.size() == steps.size() && "Mismatch in number of arguments");

  // Create the loops iteratively and store the induction variables.
  SmallVector<Value, 4> ivs;
  ivs.reserve(lbs.size());
  SmallVector<AffineForOp> loopsFromOuterToInner;
  for (unsigned i = 0, e = lbs.size(); i < e; ++i) {
    // Callback for creating the loop body, always creates the terminator.
    auto loopBody = [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                        ValueRange iterArgs) {
      ivs.push_back(iv);
      // In the innermost loop, call the body builder.
      if (i == e - 1 && bodyBuilderFn) {
        OpBuilder::InsertionGuard nestedGuard(nestedBuilder);
        SmallVector<Value> toYield =
            bodyBuilderFn(nestedBuilder, nestedLoc, ivs, iterArgs);
        nestedBuilder.create<AffineYieldOp>(nestedLoc, toYield);
      } else {
        // This loop should return the results of the next inner loop,
        // but it hasn't been created yet. Patch it up at the end.
        nestedBuilder.create<AffineYieldOp>(nestedLoc);
      }
    };

    // Delegate actual loop creation to the callback in order to dispatch
    // between constant- and variable-bound loops.
    SmallVector<Value> iterArgs =
        i == 0 ? outputInits : loopsFromOuterToInner.back().getRegionIterArgs();
    auto loop = buildAffineLoopFromValues(builder, loc, lbs[i], ubs[i],
                                          steps[i], iterArgs, loopBody);
    builder.setInsertionPointToStart(loop.getBody());
    loopsFromOuterToInner.push_back(loop);
  }

  // For all loops but the inner-most, propagate the results of the inner loop
  // to the iter args of the outer loop.
  for (int loopIndex = loopsFromOuterToInner.size() - 2; loopIndex >= 0;
       --loopIndex) {
    AffineForOp outerLoop = loopsFromOuterToInner[loopIndex];
    AffineYieldOp outerYield =
        cast<AffineYieldOp>(outerLoop.getRegion().getBlocks().front().back());
    auto innerLoopResults = loopsFromOuterToInner[loopIndex + 1].getResults();
    outerYield->setOperands(innerLoopResults);
  }

  return loopsFromOuterToInner.front();
}

// upstream Linalg/Utils/Utils.h::GenerateLoopNest forces memref semantics.
AffineForOp generateLoopNest(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes, ValueRange outputInits,
    function_ref<SmallVector<Value>(OpBuilder &, Location, ValueRange,
                                    ValueRange)>
        bodyBuilderFn) {
  SmallVector<Value, 4> lbs, ubs, steps;

  // An inlined copy of Utils.cpp::unpackRanges
  for (Range range : loopRanges) {
    lbs.emplace_back(getValueOrCreateConstantIndexOp(b, loc, range.offset));
    ubs.emplace_back(getValueOrCreateConstantIndexOp(b, loc, range.size));
    steps.emplace_back(getValueOrCreateConstantIndexOp(b, loc, range.stride));
  }

  // Affine loops require constant steps.
  SmallVector<int64_t, 4> constantSteps;
  constantSteps.reserve(steps.size());
  for (Value v : steps) {
    auto constVal = getConstantIntValue(v);
    assert(constVal.has_value() && "Affine loops require constant steps");
    constantSteps.push_back(constVal.value());
  }

  // Upstream's buildAffineLoopNest doesn't propagate any iter args, and we
  // need to ensure the result of the inner-most tensor insertion ops are
  // propagated as iter args through the entire loop nest. So we clone the
  // upstream buildAffineLoopNestImpl with loop-carried tensors.
  return buildAffineLoopNestWithCarriedIterArgs(
      b, loc, lbs, ubs, constantSteps, outputInits,
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange iterArgs) -> SmallVector<Value> {
        return bodyBuilderFn(b, loc, ivs, iterArgs);
      });
}

struct LowerLinalgGenericToLoops : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    LinalgOp linalgOp = cast<LinalgOp>(op.getOperation());
    SmallVector<Range, 4> loopRanges =
        linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
    SmallVector<utils::IteratorType> iteratorTypes =
        linalgOp.getIteratorTypesArray();

    SmallVector<Value> allIvs;

    AffineForOp outermostLoop = generateLoopNest(
        rewriter, op.getLoc(), loopRanges, linalgOp, iteratorTypes,
        op.getOutputs(),
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> SmallVector<Value> {
          allIvs.append(ivs.begin(), ivs.end());
          return emitScalarImplementation(b, loc, allIvs, op, iterArgs);
        });

    rewriter.replaceOp(op, outermostLoop);
    return success();
  }
};

struct TensorLinalgToAffineLoops
    : impl::TensorLinalgToAffineLoopsBase<TensorLinalgToAffineLoops> {
  using TensorLinalgToAffineLoopsBase::TensorLinalgToAffineLoopsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerLinalgGenericToLoops, ExpandAffineApply>(context);
    // Greedy is necessary here because LowerLinalgGenericToLoops generates
    // affine.apply ops.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
