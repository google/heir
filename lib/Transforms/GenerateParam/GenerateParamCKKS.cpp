#include <cmath>
#include <optional>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/RangeAnalysis/RangeAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/LogArithmetic.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "GenerateParamCKKS"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMCKKS
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

struct GenerateParamCKKS : impl::GenerateParamCKKSBase<GenerateParamCKKS> {
  using GenerateParamCKKSBase::GenerateParamCKKSBase;

  // In CKKS, the modulus for L0 should be larger than the
  // scaling modulus, however, the number of extra bits is often
  // empirically chosen. We use RangeAnalysis to find the
  // maximum number of extra bits needed for the L0 modulus.
  std::optional<int> getExtraBitsForLevel0() {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    // RangeAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    // For double input in range [-1, 1], we use Log2Arithmetic::of(1) to
    // represent it.
    solver.load<RangeAnalysis>(Log2Arithmetic::of(inputRange));
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    std::optional<double> extraBits;

    getOperation()->walk([&](Operation* op) {
      for (auto result : op->getResults()) {
        if (mgmt::shouldHaveMgmtAttribute(result, &solver) &&
            getLevelFromMgmtAttr(result) == 0) {
          auto range = getRange(result, &solver);
          if (range.has_value()) {
            auto resultExtraBits = range->getLog2Value();
            if (!extraBits.has_value() || resultExtraBits > extraBits.value()) {
              extraBits = resultExtraBits;
            }
          }
        }
      }
    });

    if (!extraBits.has_value()) {
      return std::nullopt;
    }
    // 2 more bits for cushion
    return ceil(extraBits.value()) + 2;
  }

  void runOnOperation() override {
    // Deal with first mod bits
    auto extraBits = getExtraBitsForLevel0();

    if (firstModBits == 0) {
      if (!extraBits.has_value()) {
        emitError(getOperation()->getLoc())
            << "Cannot generate CKKS parameters without first modulus bits "
               "or extra bits for level 0.\n";
        signalPassFailure();
        return;
      }
      firstModBits = scalingModBits + extraBits.value();
      LLVM_DEBUG(llvm::dbgs() << "First modulus bits not specified, using "
                              << firstModBits << " bits.\n");
    } else if (extraBits.has_value() &&
               firstModBits - scalingModBits < extraBits.value()) {
      emitWarning(getOperation()->getLoc())
          << "Range Analysis indicate that the first modulus must be larger "
             "than the scaling modulus by at least "
          << extraBits.value() << " bits.\n";
    }

    std::optional<int> maxLevel = getMaxLevel(getOperation());

    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<ckks::SchemeParamAttr>(
                ckks::CKKSDialect::kSchemeParamAttrName)) {
      // TODO: put this in validate-noise once CKKS noise model is in
      auto schemeParam = ckks::getSchemeParamFromAttr(schemeParamAttr);
      if (schemeParam.getLevel() < maxLevel.value_or(0)) {
        getOperation()->emitOpError()
            << "The level in the scheme param is smaller than the max level.\n";
        signalPassFailure();
        return;
      }
      return;
    }

    // for lattigo, defaults to extended encryption technique
    if (moduleIsLattigo(getOperation())) {
      encryptionTechniqueExtended = true;
    }

    auto schemeParam = ckks::SchemeParam::getConcreteSchemeParam(
        firstModBits, scalingModBits, maxLevel.value_or(0), slotNumber,
        usePublicKey, encryptionTechniqueExtended, reducedError);

    LLVM_DEBUG(llvm::dbgs() << "Scheme Param:\n" << schemeParam << "\n");

    // annotate ckks::SchemeParamAttr to ModuleOp
    getOperation()->setAttr(
        ckks::CKKSDialect::kSchemeParamAttrName,
        ckks::SchemeParamAttr::get(
            &getContext(), log2(schemeParam.getRingDim()),
            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(schemeParam.getQi())),
            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(schemeParam.getPi())),
            schemeParam.getLogDefaultScale(),
            usePublicKey ? ckks::CKKSEncryptionType::pk
                         : ckks::CKKSEncryptionType::sk,
            encryptionTechniqueExtended
                ? ckks::CKKSEncryptionTechnique::extended
                : ckks::CKKSEncryptionTechnique::standard));
  }
};

}  // namespace heir
}  // namespace mlir
