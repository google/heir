#include "lib/Dialect/Lattigo/Transforms/AllocToInPlace.h"

#include <cstddef>
#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/AllocToInPlaceUtils.h"
#include "mlir/include/mlir/Analysis/AliasAnalysis.h"      // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Liveness.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"                // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/ValueBoundsOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

namespace {

// Sets the level of a potentially newly created value.
static inline void setValueToLevel(DataFlowSolver* solver, Value value,
                                   int level) {
  auto* lattice = solver->getOrCreateState<LevelLattice>(value);
  lattice->getValue().setLevel(level);
}

}  // namespace

// Returns true if the index value ranges of the two operations are provably
// equal.
static bool areIndicesEqual(ValueRange indicesA, ValueRange indicesB) {
  if (indicesA.size() != indicesB.size()) return false;
  for (size_t i = 0; i < indicesA.size(); ++i) {
    if (indicesA[i] == indicesB[i]) continue;
    auto eq = ValueBoundsConstraintSet::areEqual(indicesA[i], indicesB[i]);
    if (failed(eq) || !*eq) return false;
  }
  return true;
}

static Value findStorageFromMemrefPattern(Operation* op, Liveness* liveness,
                                          AliasAnalysis* aliasAnalysis) {
  if (op->getNumResults() == 0) return nullptr;

  // Search for the load-op-store pattern, where:
  // 1. A value is loaded from a memref: %inputVal = memref.load
  // %memref[%indices]
  // 2. The value is used by this op.
  // 3. The result of the op is stored back to the same memref:
  //    memref.store %result, %memref[%indices]
  // If %inputVal is dead after the op, %memref can be reused in-place.
  for (auto& use : op->getResult(0).getUses()) {
    auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner());
    if (!storeOp) continue;

    Value memref = storeOp.getMemRef();
    ValueRange indices = storeOp.getIndices();
    for (Value inputVal : op->getOperands()) {
      auto loadOp = dyn_cast_or_null<memref::LoadOp>(inputVal.getDefiningOp());
      if (!loadOp) continue;

      if (!aliasAnalysis->alias(loadOp.getMemRef(), memref).isMust()) continue;

      if (!areIndicesEqual(indices, loadOp.getIndices())) continue;

      if (liveness->isDeadAfter(inputVal, op)) {
        return inputVal;
      }
    }
  }
  return nullptr;
}

template <typename BinOp, typename InPlaceOp>
struct ConvertBinOp : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  ConvertBinOp(mlir::MLIRContext* context, Liveness* liveness,
               DominanceInfo* domInfo, DataFlowSolver* solver,
               DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo,
               AliasAnalysis* aliasAnalysis)
      : OpRewritePattern<BinOp>(context),
        liveness(liveness),
        domInfo(domInfo),
        solver(solver),
        blockToStorageInfo(blockToStorageInfo),
        aliasAnalysis(aliasAnalysis) {}

  LogicalResult matchAndRewrite(BinOp op,
                                PatternRewriter& rewriter) const override {
    auto& storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage =
        storageInfo.getAvailableStorage(op, liveness, domInfo, solver);
    if (!storage) {
      storage = findStorageFromMemrefPattern(op, liveness, aliasAnalysis);
    }
    if (!storage) {
      return rewriter.notifyMatchFailure(op, "no available storage found");
    }

    // InPlaceOp has the form: output = InPlaceOp(evaluator, lhs, rhs,
    // inplace) where inplace is the actual output but for SSA form we need to
    // return a new value
    auto inplaceOp = InPlaceOp::create(
        rewriter, op.getLoc(), op.getOperand(1).getType(), op.getOperand(0),
        op.getOperand(1), op.getOperand(2), storage);

    // Update storage info, which must happen before the op is removed
    storageInfo.replaceAllocWithInPlace(op, inplaceOp, storage);
    setValueToLevel(solver, inplaceOp->getResult(0),
                    getLevel(op->getResult(0), solver).value().getInt());
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  DominanceInfo* domInfo;
  DataFlowSolver* solver;
  DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo;
  AliasAnalysis* aliasAnalysis;
};

template <typename UnaryOp, typename InPlaceOp>
struct ConvertUnaryOp : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  ConvertUnaryOp(
      mlir::MLIRContext* context, Liveness* liveness, DominanceInfo* domInfo,
      DataFlowSolver* solver,
      DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo,
      AliasAnalysis* /*aliasAnalysis*/)
      : OpRewritePattern<UnaryOp>(context),
        liveness(liveness),
        domInfo(domInfo),
        solver(solver),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter& rewriter) const override {
    auto& storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage =
        storageInfo.getAvailableStorage(op, liveness, domInfo, solver);
    if (!storage) {
      return rewriter.notifyMatchFailure(op, "no available storage found");
    }

    // InPlaceOp has the form: output = InPlaceOp(evaluator, lhs, inplace)
    // where inplace is the actual output but for SSA form we need to return a
    // new value
    auto inplaceOp =
        InPlaceOp::create(rewriter, op.getLoc(), op.getOperand(1).getType(),
                          op.getOperand(0), op.getOperand(1), storage);

    storageInfo.replaceAllocWithInPlace(op, inplaceOp, storage);
    setValueToLevel(solver, inplaceOp->getResult(0),
                    getLevel(op->getResult(0), solver).value().getInt());
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  DominanceInfo* domInfo;
  DataFlowSolver* solver;
  DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo;
};

template <typename RotateOp, typename InPlaceOp>
struct ConvertRotateOp : public OpRewritePattern<RotateOp> {
  using OpRewritePattern<RotateOp>::OpRewritePattern;

  ConvertRotateOp(
      mlir::MLIRContext* context, Liveness* liveness, DominanceInfo* domInfo,
      DataFlowSolver* solver,
      DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo,
      AliasAnalysis* aliasAnalysis)
      : OpRewritePattern<RotateOp>(context),
        liveness(liveness),
        domInfo(domInfo),
        solver(solver),
        blockToStorageInfo(blockToStorageInfo),
        aliasAnalysis(aliasAnalysis) {}

  LogicalResult matchAndRewrite(RotateOp op,
                                PatternRewriter& rewriter) const override {
    auto& storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage =
        storageInfo.getAvailableStorage(op, liveness, domInfo, solver);
    if (!storage) {
      storage = findStorageFromMemrefPattern(op, liveness, aliasAnalysis);
    }
    if (!storage) {
      return rewriter.notifyMatchFailure(op, "no available storage found");
    }

    // InPlaceOp has the form: output = InPlaceOp(evaluator, input, inplace,
    // dynamic_shift?, static_shift?) where inplace is the actual output but for
    // SSA form we need to return a new value. Handle both dynamic_shift SSA
    // value and static_shift attribute.
    Value dynamicShift = op.getDynamicShift();
    IntegerAttr staticShift = op.getStaticShiftAttr();
    if (!staticShift && !dynamicShift) {
      return rewriter.notifyMatchFailure(
          op, "rotate op must have either static or dynamic shift");
    }
    auto inplaceOp =
        staticShift
            ? InPlaceOp::create(rewriter, op.getLoc(), op.getInput().getType(),
                                op.getEvaluator(), op.getInput(), storage,
                                /*dynamic_shift=*/nullptr, staticShift)
            : InPlaceOp::create(rewriter, op.getLoc(), op.getInput().getType(),
                                op.getEvaluator(), op.getInput(), storage,
                                dynamicShift,
                                /*static_shift=*/nullptr);

    // update storage info
    storageInfo.replaceAllocWithInPlace(op, inplaceOp, storage);
    setValueToLevel(solver, inplaceOp->getResult(0),
                    getLevel(op->getResult(0), solver).value().getInt());
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  DominanceInfo* domInfo;
  DataFlowSolver* solver;
  DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo;
  AliasAnalysis* aliasAnalysis;
};

template <typename DropLevelOp, typename InPlaceOp>
struct ConvertDropLevelOp : public OpRewritePattern<DropLevelOp> {
  using OpRewritePattern<DropLevelOp>::OpRewritePattern;

  ConvertDropLevelOp(
      mlir::MLIRContext* context, Liveness* liveness, DominanceInfo* domInfo,
      DataFlowSolver* solver,
      DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo,
      AliasAnalysis* /*aliasAnalysis*/)
      : OpRewritePattern<DropLevelOp>(context),
        liveness(liveness),
        domInfo(domInfo),
        solver(solver),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(DropLevelOp op,
                                PatternRewriter& rewriter) const override {
    auto& storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage =
        storageInfo.getAvailableStorage(op, liveness, domInfo, solver);
    if (!storage) {
      return rewriter.notifyMatchFailure(op, "no available storage found");
    }

    // InPlaceOp has the form: output = InPlaceOp(evaluator, lhs, inplace)
    // {levelToDrop} where inplace is the actual output but for SSA form we need
    // to return a new value
    auto inplaceOp = InPlaceOp::create(
        rewriter, op.getLoc(), op.getOperand(1).getType(), op.getOperand(0),
        op.getOperand(1), storage, op.getLevelToDrop());

    // update storage info
    storageInfo.replaceAllocWithInPlace(op, inplaceOp, storage);
    setValueToLevel(solver, inplaceOp->getResult(0),
                    getLevel(op->getResult(0), solver).value().getInt());
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  DominanceInfo* domInfo;
  DataFlowSolver* solver;
  DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo;
};

#define GEN_PASS_DEF_ALLOCTOINPLACE
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

struct AllocToInPlace : impl::AllocToInPlaceBase<AllocToInPlace> {
  using AllocToInPlaceBase::AllocToInPlaceBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }
    Liveness liveness(getOperation());
    AliasAnalysis& aliasAnalysis = getAnalysis<AliasAnalysis>();
    DominanceInfo domInfo(getOperation());

    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    DenseMap<Block*, CallerProvidedStorageInfo> blockToStorageInfo =
        initializeAllocToInPlaceBlockStorage<RLWECiphertextType>(
            getOperation());

    patterns.add<
        // BGV
        ConvertBinOp<lattigo::BGVAddNewOp, lattigo::BGVAddOp>,
        ConvertBinOp<lattigo::BGVSubNewOp, lattigo::BGVSubOp>,
        ConvertBinOp<lattigo::BGVMulNewOp, lattigo::BGVMulOp>,
        ConvertUnaryOp<lattigo::BGVRelinearizeNewOp, lattigo::BGVRelinearizeOp>,
        ConvertUnaryOp<lattigo::BGVRescaleNewOp, lattigo::BGVRescaleOp>,
        ConvertRotateOp<lattigo::BGVRotateColumnsNewOp,
                        lattigo::BGVRotateColumnsOp>,
        // CKKS
        ConvertBinOp<lattigo::CKKSAddNewOp, lattigo::CKKSAddOp>,
        ConvertBinOp<lattigo::CKKSSubNewOp, lattigo::CKKSSubOp>,
        ConvertBinOp<lattigo::CKKSMulNewOp, lattigo::CKKSMulOp>,
        ConvertUnaryOp<lattigo::CKKSRelinearizeNewOp,
                       lattigo::CKKSRelinearizeOp>,
        ConvertUnaryOp<lattigo::CKKSRescaleNewOp, lattigo::CKKSRescaleOp>,
        ConvertRotateOp<lattigo::CKKSRotateNewOp, lattigo::CKKSRotateOp>,
        // RLWE
        ConvertUnaryOp<lattigo::RLWENegateNewOp, lattigo::RLWENegateOp>,
        ConvertDropLevelOp<lattigo::RLWEDropLevelNewOp,
                           lattigo::RLWEDropLevelOp>>(
        context, &liveness, &domInfo, &solver, &blockToStorageInfo,
        &aliasAnalysis);

    // The greedy policy relies on the order of processing the operations.
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
