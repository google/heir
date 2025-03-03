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
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

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

  void runOnOperation() override {
    auto bgvSchemeParamAttr = mlir::dyn_cast<bgv::SchemeParamAttr>(
        getOperation()->getAttr(bgv::BGVDialect::kSchemeParamAttrName));
    auto Q = bgvSchemeParamAttr.getQ();
    std::vector<int64_t> qi;
    qi.reserve(Q.size());
    for (int i = 0; i < Q.size(); i++) {
      qi.push_back(Q[i]);
    }

    auto t = bgvSchemeParamAttr.getPlaintextModulus();

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

    auto getI64Attr = [&](int64_t value) -> IntegerAttr {
      return IntegerAttr::get(
          IntegerType::get(getOperation()->getContext(), 64), value);
    };

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

    // this part is kind of "back propagation" of scales
    //
    // for adjust scale op, find scale from arith op
    // the sequence is adjust_scale + mod_reduce + arith
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
      auto nextOp = dyn_cast<mgmt::ModReduceOp>(op->use_begin()->getOwner());
      if (!nextOp) {
        getOperation()->emitError(
            "AdjustScaleOp does not have ModReduceOp as its use");
      }
      if (!nextOp->hasOneUse()) {
        getOperation()->emitError("ModReduceOp does not have exactly one use");
      }
      auto *arithOp = nextOp->use_begin()->getOwner();
      if (!mlir::isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(arithOp)) {
        getOperation()->emitError(
            "ModReduceOp does not have MulIOp, AddIOp, or SubIOp as its use");
      }

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

      auto level = mgmtAttr.getLevel();
      int64_t newScale = qi[level] * arithScale % t;
      op->setAttr("scale", getI64Attr(newScale));

      // compute the constant needed to adjust the scale for mul constant
      auto input = op->getOperand(0);
      auto inputScale = lookupScale(input);
      if (inputScale == -1) {
        getOperation()->emitError("Input does not have scale");
      }
      auto inputScaleInverse =
          heir::polynomial::multiplicativeInverse(llvm::APInt(64, inputScale),
                                                  llvm::APInt(64, t))
              .getSExtValue();
      auto deltaScale = (newScale * inputScaleInverse) % t;
      op->setAttr("delta_scale", getI64Attr(deltaScale));
    });

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

        // validate the input scale is the same
        if (op->getNumOperands() > 1) {
          auto scale = 0;
          for (auto operand : op->getOperands()) {
            auto operandScale = lookupScale(operand);
            if (operandScale == -1) {
              continue;
            }
            if (scale == 0) {
              scale = operandScale;
            } else if (scale != operandScale) {
              op->emitError("Different scales");
            }
          }
        }
      });
    });
  }
};

}  // namespace heir
}  // namespace mlir
