#include "lib/Transforms/PopulateScale/PopulateScale.h"

#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "PopulateScale"

namespace mlir::heir::polynomial {
llvm::APInt multiplicativeInverse(const llvm::APInt &x,
                                  const llvm::APInt &modulo);
}  // namespace mlir::heir::polynomial

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_POPULATESCALE
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

struct PopulateScale : impl::PopulateScaleBase<PopulateScale> {
  using PopulateScaleBase::PopulateScaleBase;

  // ciphetext modulus of each level
  std::vector<int64_t> qi;
  // plaintext modulus
  int64_t t;

  void populateScaleForAdjustScaleOp(
      std::function<int64_t(Value)> lookupScale) {
    auto arithOpGetOperandScale = [&](Operation *arithOp) {
      int64_t arithScale = 0;
      for (auto operand : arithOp->getOperands()) {
        auto operandScale = lookupScale(operand);
        if (operandScale == -1) {
          continue;
        }
        arithScale = operandScale;
        break;
      }
      if (arithScale == 0) {
        getOperation()->emitError("ArithOp does not have scale");
      }
      return arithScale;
    };

    // this part is kind of "back propagation" of scales
    //
    // for adjust scale op, find scale from arith op
    // the sequence is adjust_scale (+ mod_reduce) + arith
    getOperation()->walk([&](mgmt::AdjustScaleOp op) {
      auto mgmtAttr = op->getAttrOfType<mgmt::MgmtAttr>(
          mgmt::MgmtDialect::kArgMgmtAttrName);
      if (!mgmtAttr) {
        getOperation()->emitError("AdjustScaleOp does not have MgmtAttr");
      }
      if (!op->hasOneUse()) {
        getOperation()->emitError(
            "AdjustScaleOp does not have exactly one use");
      }
      auto *nextOp = op->use_begin()->getOwner();

      Operation *arithOp = nullptr;
      bool nextIsModReduceOp = isa<mgmt::ModReduceOp>(nextOp);
      if (auto modReduceOp = dyn_cast<mgmt::ModReduceOp>(nextOp)) {
        if (!modReduceOp->hasOneUse()) {
          getOperation()->emitError(
              "ModReduceOp does not have exactly one use");
        }
        arithOp = modReduceOp->use_begin()->getOwner();
      } else {
        arithOp = nextOp;
      }
      if (!mlir::isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(arithOp)) {
        getOperation()->emitError(
            "AdjustScaleOp does not have MulIOp, AddIOp, or SubIOp as its "
            "(subsequent) use");
      }

      auto arithScale = arithOpGetOperandScale(arithOp);

      int64_t scale = arithScale;
      if (nextIsModReduceOp) {
        auto level = mgmtAttr.getLevel();
        scale = qi[level] * scale % t;
      }
      op.setScale(scale);
    });
  }

  struct ConvertAdjustScaleToMulPlain
      : public OpRewritePattern<mgmt::AdjustScaleOp> {
    using OpRewritePattern<mgmt::AdjustScaleOp>::OpRewritePattern;

    ConvertAdjustScaleToMulPlain(MLIRContext *context, int64_t plaintextModulus)
        : OpRewritePattern<mgmt::AdjustScaleOp>(context, /*benefit=*/1),
          plaintextModulus(plaintextModulus) {}

    LogicalResult matchAndRewrite(mgmt::AdjustScaleOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputScale = mgmt::getMgmtAttrFromValue(op.getInput()).getScale();
      auto scale = op.getScale();
      // no need to adjust scale
      if (scale == inputScale) {
        rewriter.replaceAllOpUsesWith(op, op->getOperand(0));
        return success();
      }

      auto inputScaleInverse =
          heir::polynomial::multiplicativeInverse(
              llvm::APInt(64, inputScale), llvm::APInt(64, plaintextModulus))
              .getSExtValue();
      auto deltaScale = (scale * inputScaleInverse) % plaintextModulus;

      // lower to (input * all_ones)

      auto mgmtAttr = op->getAttrOfType<mgmt::MgmtAttr>(
          mgmt::MgmtDialect::kArgMgmtAttrName);

      auto inputType = op.getInput().getType();

      APInt one(getElementTypeOrSelf(inputType).getIntOrFloatBitWidth(), 1);
      TypedAttr constantAttr;
      if (auto inputTensorType = dyn_cast<RankedTensorType>(inputType)) {
        constantAttr = DenseElementsAttr::get(inputTensorType, one);
      } else {
        constantAttr = IntegerAttr::get(inputType, one);
      }

      // create arith.constant at the beginning of the function
      auto funcOp = op->getParentOfType<func::FuncOp>();
      rewriter.setInsertionPointToStart(&funcOp.getBody().front());
      auto allOnes = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), inputType, constantAttr);
      auto allOnesMgmtAttr =
          mgmt::MgmtAttr::get(op->getContext(), mgmtAttr.getLevel(),
                              mgmtAttr.getDimension(), deltaScale);
      // do not annotate it to arith.constant as canonicalizer will merge
      // arith.constant with same value and mgmt attr will be lost

      rewriter.setInsertionPoint(op);
      // no-op also for preventing mul 1 being constant folded
      auto noOp = rewriter.create<mgmt::NoOp>(op.getLoc(), inputType,
                                              allOnes.getResult());
      noOp->setAttr(mgmt::MgmtDialect::kArgMgmtAttrName, allOnesMgmtAttr);
      auto mulOp = rewriter.create<arith::MulIOp>(
          op.getLoc(), inputType, op.getInput(), noOp.getOutput());
      auto mulOpMgmtAttr =
          mgmt::MgmtAttr::get(op->getContext(), mgmtAttr.getLevel(),
                              mgmtAttr.getDimension(), scale);
      mulOp->setAttr(mgmt::MgmtDialect::kArgMgmtAttrName, mulOpMgmtAttr);
      rewriter.replaceAllOpUsesWith(op, mulOp.getResult());
      return success();
    }

   private:
    int64_t plaintextModulus;
  };

  void runOnOperation() override {
    auto bgvSchemeParamAttr = mlir::dyn_cast<bgv::SchemeParamAttr>(
        getOperation()->getAttr(bgv::BGVDialect::kSchemeParamAttrName));
    auto Q = bgvSchemeParamAttr.getQ();
    qi.reserve(Q.size());
    for (int i = 0; i < Q.size(); i++) {
      qi.push_back(Q[i]);
    }

    t = bgvSchemeParamAttr.getPlaintextModulus();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    // ScaleAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    // set input scale to 1
    // NOTE that this is important for the input level for both
    // before-mul,include-mul-first={true,false} style mgmt
    solver.load<ScaleAnalysis>(
        bgv::SchemeParam::getSchemeParamFromAttr(bgvSchemeParamAttr),
        /*inputScale*/ 1);

    // the first analysis partially populates the scales
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    auto lookupScale = [&](Value value) -> int64_t {
      const auto *scaleLattice = solver.lookupState<ScaleLattice>(value);
      if (!scaleLattice) {
        return -1;
      }
      auto scaleState = scaleLattice->getValue();
      if (!scaleState.isInitialized()) {
        return -1;
      }
      return scaleState.getScale();
    };

    // populate scales for adjust scale op
    populateScaleForAdjustScaleOp(lookupScale);

    // re-run the analysis to propagate all scales
    solver.eraseAllStates();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    auto updateMgmtAttr = [&](Operation *op) {
      auto mgmtAttr =
          dyn_cast_or_null<mgmt::MgmtAttr>(op->getAttrOfType<mgmt::MgmtAttr>(
              mgmt::MgmtDialect::kArgMgmtAttrName));
      if (!mgmtAttr) {
        return;
      }
      auto scale = lookupScale(op->getResult(0));
      if (scale == -1) {
        return;
      }
      auto newMgmtAttr =
          mgmt::MgmtAttr::get(op->getContext(), mgmtAttr.getLevel(),
                              mgmtAttr.getDimension(), scale);
      op->setAttr(mgmt::MgmtDialect::kArgMgmtAttrName, newMgmtAttr);
    };

    // annotate scale
    getOperation()->walk([&](secret::GenericOp genericOp) {
      auto funcOp = genericOp->getParentOfType<func::FuncOp>();
      OpBuilder b = OpBuilder::atBlockBegin(&funcOp.getBody().front());

      Block *body = genericOp.getBody();
      for (auto i = 0; i != body->getNumArguments(); ++i) {
        auto blockArg = body->getArgument(i);
        auto mgmtAttr = dyn_cast_or_null<mgmt::MgmtAttr>(
            genericOp.getArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName));
        if (!mgmtAttr) {
          continue;
        }
        auto newMgmtAttr =
            mgmt::MgmtAttr::get(genericOp->getContext(), mgmtAttr.getLevel(),
                                mgmtAttr.getDimension(), lookupScale(blockArg));
        genericOp.setArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName,
                             newMgmtAttr);
      }
      genericOp.walk([&](Operation *op) {
        if (op->getNumResults() != 1 || mlir::isa<secret::GenericOp>(op)) {
          return;
        }
        updateMgmtAttr(op);

        // update the scale for ct-pt op
        if (mlir::isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(op)) {
          int64_t scale = -1;
          int plaintextOperandIndex = -1;
          // find the plaintext operand and get mgmt attr from the other
          // operand
          for (auto i = 0; i != op->getNumOperands(); ++i) {
            auto operandScale = lookupScale(op->getOperand(i));
            if (operandScale == -1) {
              plaintextOperandIndex = i;
            } else {
              scale = operandScale;
            }
          }
          if (scale == -1) {
            op->emitError("ArithOp does not have scale");
          }
          if (plaintextOperandIndex != -1) {
            auto plaintextOperand = op->getOperand(plaintextOperandIndex);
            auto noOp = mlir::dyn_cast_or_null<mgmt::NoOp>(
                plaintextOperand.getDefiningOp());
            auto arithConstantOp = mlir::dyn_cast_or_null<arith::ConstantOp>(
                plaintextOperand.getDefiningOp());
            if (noOp) {
              auto mgmtAttr = noOp->getAttrOfType<mgmt::MgmtAttr>(
                  mgmt::MgmtDialect::kArgMgmtAttrName);
              if (!mgmtAttr) {
                op->emitError("NoOp does not have MgmtAttr");
              } else {
                auto newMgmtAttr =
                    mgmt::MgmtAttr::get(noOp->getContext(), mgmtAttr.getLevel(),
                                        mgmtAttr.getDimension(), scale);
                noOp->setAttr(mgmt::MgmtDialect::kArgMgmtAttrName, newMgmtAttr);
                // set the lattice for later validation
                auto *lattice =
                    solver.getOrCreateState<ScaleLattice>(noOp.getResult());
                (void)lattice->join(ScaleState(scale));
              }
            } else if (arithConstantOp) {
              // create a new arith.constant with mgmt attr
              // this is because an arith.constant op can be used in multiple
              // places and we don't want to change the original one
              auto newArithConstantOp = b.create<arith::ConstantOp>(
                  arithConstantOp.getLoc(), arithConstantOp.getType(),
                  arithConstantOp.getValue());
              auto mgmtAttr = arithConstantOp->getAttrOfType<mgmt::MgmtAttr>(
                  mgmt::MgmtDialect::kArgMgmtAttrName);
              if (!mgmtAttr) {
                op->emitError("ArithConstantOp does not have MgmtAttr");
              } else {
                auto newMgmtAttr = mgmt::MgmtAttr::get(
                    arithConstantOp->getContext(), mgmtAttr.getLevel(),
                    mgmtAttr.getDimension(), scale);
                newArithConstantOp->setAttr(mgmt::MgmtDialect::kArgMgmtAttrName,
                                            newMgmtAttr);
                op->setOperand(plaintextOperandIndex,
                               newArithConstantOp.getResult());
                // set the lattice for later validation
                auto *lattice = solver.getOrCreateState<ScaleLattice>(
                    newArithConstantOp.getResult());
                (void)lattice->join(ScaleState(scale));
              }
            } else {
              op->emitWarning() << "plaintext operand is not defined by "
                                   "arith.constant or mgmt.no_op, could "
                                   "not annotate scale in mgmt attr.";
            }
          }
        }

        // validate the input scale is the same
        if (op->getNumOperands() > 1) {
          auto scale = 0;
          for (auto operand : op->getOperands()) {
            auto operandScale = lookupScale(operand);
            if (scale == 0) {
              scale = operandScale;
            } else if (scale != operandScale) {
              op->emitError("Different scales");
            }
          }
        }
      });
    });

    // convert adjust scale to mul plain
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertAdjustScaleToMulPlain>(&getContext(), t);
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    // run canonicalizer and CSE to clean up arith.constant and move no-op out
    // of the loop
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
