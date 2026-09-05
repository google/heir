#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Transforms/PopulateScale/PopulateScalePatterns.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"       // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
// IWYU pragma: end_keep

#define DEBUG_TYPE "populate-scale-ckks"

namespace mlir {
namespace heir {

class CKKSAdjustScaleMaterializer : public AdjustScaleMaterializer {
 public:
  virtual ~CKKSAdjustScaleMaterializer() = default;

  int64_t deltaScale(int64_t scale, int64_t inputScale) const override {
    // TODO(#1640): support high-precision scale management
    return scale - inputScale;
  }
};

#define GEN_PASS_DEF_POPULATESCALECKKS
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

LogicalResult createAndRunDataflow(Operation* op, DataFlowSolver& solver,
                                   int64_t logDefaultScale,
                                   ckks::SchemeParamAttr ckksSchemeParamAttr,
                                   bool beforeMulIncludeFirstMul,
                                   bool assumeTargetScaleForMul = false) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  auto inputScale = logDefaultScale;
  if (beforeMulIncludeFirstMul) {
    LDBG() << "Encoding at scale^2 due to 'include-first-mul' config";
    inputScale *= 2;
  }
  auto param = ckks::getSchemeParamFromAttr(ckksSchemeParamAttr);
  solver.load<ScaleAnalysis<CKKSScaleModel>>(param,
                                             /*inputScale*/ inputScale,
                                             assumeTargetScaleForMul);

  return solver.initializeAndRun(op);
}

struct MergeAdjustScaleIntoInit : public OpRewritePattern<mgmt::AdjustScaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mgmt::AdjustScaleOp op,
                                PatternRewriter& rewriter) const override {
    Value input = op.getInput();
    auto initOp = input.getDefiningOp<mgmt::InitOp>();
    if (!initOp) {
      return rewriter.notifyMatchFailure(op,
                                         "input is not defined by mgmt.init");
    }

    if (!initOp.getResult().hasOneUse()) {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(initOp);
      auto clonedInitOp = cast<mgmt::InitOp>(rewriter.clone(*initOp));

      rewriter.modifyOpInPlace(
          op, [&]() { op.getInputMutable().assign(clonedInitOp.getResult()); });

      initOp = clonedInitOp;
    }

    auto targetMgmtAttr = mgmt::findMgmtAttrAssociatedWith(op.getResult());
    if (!targetMgmtAttr || targetMgmtAttr.getScale() == -1) {
      return rewriter.notifyMatchFailure(
          op, "adjust_scale result has no target scale");
    }

    auto initMgmtAttr = mgmt::findMgmtAttrAssociatedWith(initOp.getResult());
    assert(initMgmtAttr && "mgmt.init result must have MgmtAttr");
    auto newMgmtAttr =
        mgmt::getMgmtAttrWithNewScale(initMgmtAttr, targetMgmtAttr.getScale());

    rewriter.modifyOpInPlace(initOp, [&]() {
      mgmt::setMgmtAttrAssociatedWith(initOp.getResult(), newMgmtAttr);
    });

    rewriter.replaceOp(op, initOp.getResult());
    return success();
  }
};

static std::optional<int64_t> getScaleOrNull(Value value,
                                             DataFlowSolver* solver) {
  if (!isBlockLive(value.getParentBlock(), solver)) return std::nullopt;
  auto* lattice = solver->lookupState<ScaleLattice<CKKSScaleModel>>(value);
  if (!lattice || !lattice->getValue().isInitialized()) return std::nullopt;
  return lattice->getValue().getScale();
}

struct PopulateScaleCKKS : impl::PopulateScaleCKKSBase<PopulateScaleCKKS> {
  using PopulateScaleCKKSBase::PopulateScaleCKKSBase;

  void runOnOperation() override {
    auto schemeParamAttr =
        getOperation()->getAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      signalPassFailure();
      return;
    }
    auto ckksSchemeParamAttr =
        mlir::dyn_cast<ckks::SchemeParamAttr>(schemeParamAttr);
    if (!ckksSchemeParamAttr) {
      getOperation()->emitError()
          << "ckks.schemeParam attribute is not a SchemeParamAttr";
      signalPassFailure();
      return;
    }
    auto logDefaultScale = ckksSchemeParamAttr.getLogDefaultScale();
    auto param = ckks::getSchemeParamFromAttr(ckksSchemeParamAttr);

    // Step 1: Run Solver 1 (Forward Analysis only)
    DataFlowSolver solver;
    if (failed(createAndRunDataflow(getOperation(), solver, logDefaultScale,
                                    ckksSchemeParamAttr,
                                    beforeMulIncludeFirstMul,
                                    /*assumeTargetScaleForMul*/ true))) {
      signalPassFailure();
      return;
    }

    // Step 2: Walk IR and Detect Mismatches -> Insert adjust_scale
    int idCounter = 0;
    getOperation()->walk([&](Operation* op) {
      if (!isBlockLive(op->getBlock(), &solver)) {
        return;
      }
      // 1. Additive & Container Ops
      if (isa<arith::AddFOp, arith::SubFOp, arith::AddIOp, arith::SubIOp,
              tensor::InsertSliceOp, tensor::InsertOp>(op)) {
        SmallVector<OpOperand*> secretOrInittedOperands;
        for (auto& operand : op->getOpOperands()) {
          if (isSecret(operand.get(), &solver) ||
              isa_and_nonnull<mgmt::InitOp>(operand.get().getDefiningOp())) {
            secretOrInittedOperands.push_back(&operand);
          }
        }
        if (secretOrInittedOperands.size() > 1) {
          bool anyUninit = false;
          int64_t maxScale = -1;
          for (auto* operand : secretOrInittedOperands) {
            auto s = getScaleOrNull(operand->get(), &solver);
            if (!s) {
              anyUninit = true;
              break;
            }
            maxScale = std::max(maxScale, *s);
          }
          if (anyUninit) return;
          for (auto* operand : secretOrInittedOperands) {
            int64_t scale = *getScaleOrNull(operand->get(), &solver);
            if (scale < maxScale) {
              OpBuilder builder(op);
              auto adjustOp = mgmt::AdjustScaleOp::create(
                  builder, op->getLoc(), operand->get(),
                  builder.getI64IntegerAttr(idCounter++));

              auto operandMgmtAttr =
                  mgmt::findMgmtAttrAssociatedWith(operand->get());
              assert(operandMgmtAttr && "operand must have MgmtAttr");
              auto newMgmtAttr =
                  mgmt::getMgmtAttrWithNewScale(operandMgmtAttr, maxScale);
              mgmt::setMgmtAttrAssociatedWith(adjustOp.getResult(),
                                              newMgmtAttr);

              operand->set(adjustOp.getResult());
            }
          }
        }
      }

      // 2. Multiplication
      if (auto mulOp = dyn_cast<arith::MulFOp>(op)) {
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        if (isSecret(lhs, &solver) || isSecret(rhs, &solver)) {
          auto sLhs = getScaleOrNull(lhs, &solver);
          auto sRhs = getScaleOrNull(rhs, &solver);
          if (!sLhs || !sRhs) return;
          int64_t scaleLhs = *sLhs;
          int64_t scaleRhs = *sRhs;

          auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(mulOp.getResult());
          assert(mgmtAttr && "mul result must have MgmtAttr");
          auto level = mgmtAttr.getLevel();

          int64_t logqi_level = logDefaultScale;
          const auto& logqi = param.getLogqi();
          if (level >= 0 && level < static_cast<int>(logqi.size())) {
            logqi_level = static_cast<int64_t>(std::llround(logqi[level]));
          }

          int64_t targetSum = logDefaultScale + logqi_level;
          if (scaleLhs + scaleRhs < targetSum) {
            int64_t targetScale;
            Value toScale;
            if (lhs == rhs) {
              toScale = lhs;
              targetScale = (targetSum + 1) / 2;
            } else {
              toScale = scaleLhs < scaleRhs ? lhs : rhs;
              int64_t scaleOther = toScale == lhs ? scaleRhs : scaleLhs;
              targetScale = targetSum - scaleOther;
            }

            OpBuilder builder(op);
            auto adjustOp = mgmt::AdjustScaleOp::create(
                builder, op->getLoc(), toScale,
                builder.getI64IntegerAttr(idCounter++));

            auto toScaleMgmtAttr = mgmt::findMgmtAttrAssociatedWith(toScale);
            assert(toScaleMgmtAttr && "operand to scale must have MgmtAttr");
            auto newMgmtAttr =
                mgmt::getMgmtAttrWithNewScale(toScaleMgmtAttr, targetScale);
            mgmt::setMgmtAttrAssociatedWith(adjustOp.getResult(), newMgmtAttr);

            op->replaceUsesOfWith(toScale, adjustOp.getResult());
          }
        }
      }

      // 2b. ModReduce
      if (auto modReduceOp = dyn_cast<mgmt::ModReduceOp>(op)) {
        Value input = modReduceOp.getInput();
        if (isSecret(input, &solver)) {
          auto sInput = getScaleOrNull(input, &solver);
          if (!sInput) return;
          int64_t scale = *sInput;
          auto inputMgmtAttr = mgmt::findMgmtAttrAssociatedWith(input);
          assert(inputMgmtAttr && "input must have MgmtAttr");
          auto level = inputMgmtAttr.getLevel();

          int64_t logqi_level = logDefaultScale;
          const auto& logqi = param.getLogqi();
          if (level >= 0 && level < static_cast<int>(logqi.size())) {
            logqi_level = static_cast<int64_t>(std::llround(logqi[level]));
          }

          int64_t newScale = scale - logqi_level;
          if (newScale < logDefaultScale) {
            int64_t targetScale = logDefaultScale + logqi_level;
            OpBuilder builder(op);
            auto adjustOp = mgmt::AdjustScaleOp::create(
                builder, op->getLoc(), input,
                builder.getI64IntegerAttr(idCounter++));

            auto newMgmtAttr =
                mgmt::getMgmtAttrWithNewScale(inputMgmtAttr, targetScale);
            mgmt::setMgmtAttrAssociatedWith(adjustOp.getResult(), newMgmtAttr);

            op->replaceUsesOfWith(input, adjustOp.getResult());
          }
        }
      }

      // 3. RegionBranchOpInterface
      if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op)) {
        mlir::RegionBranchInverseSuccessorMapping inverseMapping;
        branchOp.getSuccessorInputOperandMapping(inverseMapping);
        for (int i = 0; i < branchOp->getNumResults(); ++i) {
          Value result = branchOp->getResult(i);
          if (!isSecret(result, &solver)) continue;

          llvm::SmallVector<mlir::OpOperand*> yieldingOperands =
              inverseMapping.lookup(result);
          if (yieldingOperands.empty()) continue;

          bool anyUninit = false;
          int64_t maxScale = -1;
          for (auto* operand : yieldingOperands) {
            auto s = getScaleOrNull(operand->get(), &solver);
            if (!s) {
              anyUninit = true;
              break;
            }
            maxScale = std::max(maxScale, *s);
          }
          if (anyUninit) continue;

          for (OpOperand* operand : yieldingOperands) {
            int64_t scale = *getScaleOrNull(operand->get(), &solver);
            if (scale < maxScale) {
              Operation* terminator = operand->getOwner();
              OpBuilder builder(terminator);
              auto adjustOp = mgmt::AdjustScaleOp::create(
                  builder, terminator->getLoc(), operand->get(),
                  builder.getI64IntegerAttr(idCounter++));

              auto operandMgmtAttr =
                  mgmt::findMgmtAttrAssociatedWith(operand->get());
              assert(operandMgmtAttr && "yielding operand must have MgmtAttr");
              auto newMgmtAttr =
                  mgmt::getMgmtAttrWithNewScale(operandMgmtAttr, maxScale);
              mgmt::setMgmtAttrAssociatedWith(adjustOp.getResult(),
                                              newMgmtAttr);

              operand->set(adjustOp.getResult());
            }
          }
        }

        for (Region& region : branchOp->getRegions()) {
          for (BlockArgument blockArg : region.getArguments()) {
            if (!isSecret(blockArg, &solver)) {
              continue;
            }

            llvm::SmallVector<mlir::OpOperand*> yieldingOperands =
                inverseMapping.lookup(blockArg);
            if (yieldingOperands.empty()) continue;

            bool anyUninit = false;
            int64_t maxScale = -1;
            for (auto* operand : yieldingOperands) {
              auto s = getScaleOrNull(operand->get(), &solver);
              if (!s) {
                anyUninit = true;
                break;
              }
              maxScale = std::max(maxScale, *s);
            }
            if (anyUninit) continue;

            for (OpOperand* operand : yieldingOperands) {
              int64_t scale = *getScaleOrNull(operand->get(), &solver);
              if (scale < maxScale) {
                Operation* owner = operand->getOwner();
                OpBuilder builder(owner);
                auto adjustOp = mgmt::AdjustScaleOp::create(
                    builder, owner->getLoc(), operand->get(),
                    builder.getI64IntegerAttr(idCounter++));

                auto operandMgmtAttr =
                    mgmt::findMgmtAttrAssociatedWith(operand->get());
                assert(operandMgmtAttr &&
                       "yielding operand must have MgmtAttr");
                auto newMgmtAttr =
                    mgmt::getMgmtAttrWithNewScale(operandMgmtAttr, maxScale);
                mgmt::setMgmtAttrAssociatedWith(adjustOp.getResult(),
                                                newMgmtAttr);

                operand->set(adjustOp.getResult());
              }
            }
          }
        }
      }

      // 4. BranchOpInterface
      if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
        for (unsigned succIdx = 0;
             succIdx < branchOp->getBlock()->getNumSuccessors(); ++succIdx) {
          Block* successor = branchOp->getBlock()->getSuccessor(succIdx);
          SuccessorOperands successorOperands =
              branchOp.getSuccessorOperands(succIdx);
          for (unsigned i = 0; i < successorOperands.size(); ++i) {
            Value forwardedVal = successorOperands[i];
            if (!isSecret(forwardedVal, &solver)) continue;

            BlockArgument blockArg = successor->getArgument(i);
            auto sTarget = getScaleOrNull(blockArg, &solver);
            auto sScale = getScaleOrNull(forwardedVal, &solver);
            if (!sTarget || !sScale) continue;
            int64_t targetScale = *sTarget;
            int64_t scale = *sScale;
            if (scale < targetScale) {
              OpBuilder builder(op);
              auto adjustOp = mgmt::AdjustScaleOp::create(
                  builder, op->getLoc(), forwardedVal,
                  builder.getI64IntegerAttr(idCounter++));

              auto valMgmtAttr = mgmt::findMgmtAttrAssociatedWith(forwardedVal);
              assert(valMgmtAttr && "forwarded value must have MgmtAttr");
              auto newMgmtAttr =
                  mgmt::getMgmtAttrWithNewScale(valMgmtAttr, targetScale);
              mgmt::setMgmtAttrAssociatedWith(adjustOp.getResult(),
                                              newMgmtAttr);

              unsigned opIdx = successorOperands.getOperandIndex(i);
              op->setOperand(opIdx, adjustOp.getResult());
            }
          }
        }
      }
    });

    // An adjust_scale neither pass could determine is underdetermined by
    // design (it exists so the compiler may choose): this happens when a
    // join's operands BOTH arrive through adjust_scale chains, blocking
    // forward propagation on both sides with no downstream anchor to seed
    // the backward pass before the next bootstrap. Resolve it to its
    // canonical target — the scale that makes its paired modreduce output
    // the default scale (logDefaultScale + logq at that level), or the
    // default scale itself when unpaired — by pinning the scale on its
    // mgmt attribute; the forward analysis honors pinned adjust_scale
    // scales, so the solver2 re-run below flows through consistently and
    // ConvertAdjustScaleToMulPlain materializes the deltas.
    SmallVector<mgmt::AdjustScaleOp> unpopulatedAdjustScales;
    getOperation()->walk([&](mgmt::AdjustScaleOp op) {
      if (!isBlockLive(op->getBlock(), &solver)) {
        return;
      }
      auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(op.getResult());
      if (!mgmtAttr || mgmtAttr.getScale() == -1) {
        unpopulatedAdjustScales.push_back(op);
      }
    });
    if (!unpopulatedAdjustScales.empty()) {
      auto schemeParam = ckks::getSchemeParamFromAttr(ckksSchemeParamAttr);
      const auto& logqi = schemeParam.getLogqi();
      for (mgmt::AdjustScaleOp op : unpopulatedAdjustScales) {
        auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(op.getResult());
        if (!mgmtAttr) {
          op.emitOpError() << "unresolved adjust_scale without mgmt attr";
          signalPassFailure();
          return;
        }
        int64_t target = logDefaultScale;
        if (op->hasOneUse() &&
            isa<mgmt::ModReduceOp>(*op.getResult().getUsers().begin())) {
          int64_t level = mgmtAttr.getLevel();
          int64_t logQ =
              (level >= 0 && level < static_cast<int64_t>(logqi.size()))
                  ? static_cast<int64_t>(std::llround(logqi[level]))
                  : logDefaultScale;
          target = logDefaultScale + logQ;
        }
        mgmt::setMgmtAttrAssociatedWith(
            op.getResult(), mgmt::getMgmtAttrWithNewScale(mgmtAttr, target));
      }
    }

    // Step 3: Optimize Plaintext Scales
    RewritePatternSet optPatterns(&getContext());
    optPatterns.add<MergeAdjustScaleIntoInit>(&getContext());
    (void)walkAndApplyPatterns(getOperation(), std::move(optPatterns));

    // Step 4: Run Solver 2 (Final Propagation)
    DataFlowSolver solver2;
    if (failed(createAndRunDataflow(getOperation(), solver2, logDefaultScale,
                                    ckksSchemeParamAttr,
                                    beforeMulIncludeFirstMul))) {
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](mgmt::InitOp op) {
      if (!isBlockLive(op->getBlock(), &solver2)) {
        return;
      }
      auto* lattice =
          solver2.lookupState<ScaleLattice<CKKSScaleModel>>(op.getResult());
      if (!lattice || !lattice->getValue().isInitialized()) {
        op.emitOpError() << "Dataflow analysis failed to populate scale "
                            "lattice for result\n";
        signalPassFailure();
      }
    });

    LDBG() << "Running annotate-mgmt sub-pass";
    annotateScale<CKKSScaleModel>(getOperation(), &solver2);
    OpPassManager annotateMgmt("builtin.module");
    annotateMgmt.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(annotateMgmt, getOperation());

    // Step 5: Materialization
    LDBG() << "convert adjust_scale to mul_plain";
    RewritePatternSet patterns(&getContext());
    CKKSAdjustScaleMaterializer materializer;
    // TODO(#1641): handle arith.muli in CKKS
    patterns.add<ConvertAdjustScaleToMulPlain<arith::MulFOp>>(&getContext(),
                                                              &materializer);
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    // run canonicalizer and CSE to clean up arith.constant and move no-op out
    // of the secret.generic
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
