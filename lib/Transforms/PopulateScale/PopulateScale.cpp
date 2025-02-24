#include "lib/Transforms/PopulateScale/PopulateScale.h"

#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
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

    auto L = Q.size() - 1;

    auto t = bgvSchemeParamAttr.getPlaintextModulus();

    std::vector<int64_t> scale(L + 1, 0);

    // set initial scale to 1
    // NOTE that this is important for the first level for both
    // include-mul-first={true,false} style mgmt
    scale[L] = 1;
    // calculate per-level scaling factor as we go down
    // scale[i] = scale[i+1] * scale[i+1] * q[i+1]^-1 mod t
    for (auto i = L - 1; i >= 0; i--) {
      auto q = qi[i + 1] % t;
      auto qInvT = ::mlir::heir::polynomial::multiplicativeInverse(
          llvm::APInt(64, q), llvm::APInt(64, t));
      auto newScale =
          ((scale[i + 1] * scale[i + 1] % t) * qInvT.getSExtValue() % t);
      scale[i] = newScale;
    }

    // for arith.mul, the result scale is scale big
    std::vector<int64_t> scaleBig(L + 1, 0);
    for (auto i = 0; i <= L; i++) {
      scaleBig[i] = scale[i] * scale[i] % t;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "PopulateScale: scale = [";
      for (long i : scale) {
        llvm::dbgs() << i << ", ";
      }
      llvm::dbgs() << "]\n";
    });

    LLVM_DEBUG({
      llvm::dbgs() << "PopulateScale: scaleBig = [";
      for (long i : scaleBig) {
        llvm::dbgs() << i << ", ";
      }
      llvm::dbgs() << "]\n";
    });

    // now we calculate the scale required for adjust_scale
    // adjust_scale[i] = scale[i - 1] * scale[i - 1] * q[i] mod t
    // which means adjust_scale[i] * q[i]^-1 = scale[i - 1] * scale[i - 1] mod t
    // this is for level i - 1, the majority of the scale is scale[i - 1] *
    // scale[i - 1] namely scaleDeg = 2
    // this is highly coupled with secret-insert-mgmt-bgv!
    std::vector<int64_t> adjustScale(L + 1, 0);
    adjustScale[0] = 1;
    for (auto i = 1; i <= L; i++) {
      auto q = qi[i] % t;
      auto newScale = ((scale[i - 1] * scale[i - 1] % t) * q) % t;
      adjustScale[i] = newScale;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "PopulateScale: adjustScale = [";
      for (long i : adjustScale) {
        llvm::dbgs() << i << ", ";
      }
      llvm::dbgs() << "]\n";
    });

    getOperation()->walk([&](mgmt::AdjustScaleOp op) {
      auto mgmtAttr = op->getAttrOfType<mgmt::MgmtAttr>(
          mgmt::MgmtDialect::kArgMgmtAttrName);
      if (!mgmtAttr) {
        getOperation()->emitError("AdjustScaleOp does not have MgmtAttr");
      }
      auto level = mgmtAttr.getLevel();
      auto newScale = adjustScale[level];
      op->setAttr(
          "scale",
          IntegerAttr::get(IntegerType::get(op.getContext(), 64), newScale));
    });

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    // ScaleAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    solver.load<ScaleAnalysis>(
        bgv::SchemeParam::getSchemeParamFromAttr(bgvSchemeParamAttr));

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
