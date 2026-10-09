#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <vector>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseCanEmbModel.h"
#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Transforms/GenerateParam/GenerateParamCommon.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"          // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Transforms/GenerateParam/GenerateParam.h"
// IWYU pragma: end_keep

#define DEBUG_TYPE "GenerateParamBGV"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMBGV
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

struct GenerateParamBGV : impl::GenerateParamBGVBase<GenerateParamBGV> {
  using GenerateParamBGVBase::GenerateParamBGVBase;

  template <typename NoiseAnalysis>
  typename NoiseAnalysis::SchemeParamType generateParamByGap(
      DataFlowSolver* solver,
      const typename NoiseAnalysis::SchemeParamType& schemeParam,
      const typename NoiseAnalysis::NoiseModel& noiseModel) {
    NoiseBoundHelper<NoiseAnalysis> helper{schemeParam, noiseModel, solver};

    // for level i, the biggest gap observed.
    std::map<int, double> levelToGap;

    auto updateLevelToGap = [&](int level, double gap) {
      if (levelToGap.count(level) == 0) {
        levelToGap[level] = gap;
      } else {
        levelToGap[level] = std::max(levelToGap.at(level), gap);
      }
    };

    auto firstModSize = 0;

    getOperation()->walk([&](secret::GenericOp genericOp) {
      // gaps caused by mod reduce
      genericOp.getBody()->walk([&](mgmt::ModReduceOp op) {
        auto operandBound = helper.getBound(op.getOperand());
        auto resultBound = helper.getBound(op.getResult());
        // the gap between the operand and result
        updateLevelToGap(getLevelFromMgmtAttr(op.getOperand()).getInt(),
                         operandBound - resultBound);
        return WalkResult::advance();
      });

      // find the max noise for the first level
      genericOp.getBody()->walk([&](Operation* op) {
        for (Value result : op->getResults()) {
          if (getLevelFromMgmtAttr(result).getInt() == 0) {
            auto bound = helper.getBound(result);
            // the bound is from v_ms + v / q, where v / q is negligible
            // so originally bound(v_ms) + 1 is enough
            // after the parameter selection with smaller primes, we have
            // v_ms \approx v / q so bound(2 * v_ms) approx bound(v_ms) + 0.5
            // now we need bound(v_ms) + 1.5 or bound + 2 to ensure the noise
            firstModSize = std::max(firstModSize, 2 + int(ceil(bound)));
          }
        }
        return WalkResult::advance();
      });
    });

    auto maxLevel = levelToGap.size() + 1;
    auto qiSize = std::vector<double>(maxLevel, 0);
    qiSize[0] = firstModSize;

    for (auto& [level, gap] : levelToGap) {
      // the prime size should be larger than the gap to ensure after mod reduce
      // the noise is still within the bound
      qiSize[level] = 1 + int(ceil(gap));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Gap logqi: ";
      for (auto size : qiSize) {
        llvm::dbgs() << static_cast<int>(size) << " ";
      }
      llvm::dbgs() << "\n";
    });

    auto concreteSchemeParam =
        NoiseAnalysis::SchemeParamType::getConcreteSchemeParam(
            qiSize, schemeParam.getPlaintextModulus(), minSlotCount,
            usePublicKey, encryptionTechniqueExtended);

    return concreteSchemeParam;
  }

  template <typename NoiseModel>
  void run(const NoiseModel& model) {
    std::optional<int> maxLevel = getMaxLevel(getOperation());

    // plaintext modulus from command line option
    auto schemeParam = NoiseModel::SchemeParamType::getConservativeSchemeParam(
        maxLevel.value_or(0), plaintextModulus, minSlotCount, usePublicKey,
        encryptionTechniqueExtended);

    LLVM_DEBUG(llvm::dbgs() << "Conservative Scheme Param:\n"
                            << schemeParam << "\n");

    DataFlowSolver solver;
    if (failed(runNoiseAnalysis(getOperation(), schemeParam, model, solver))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    // use previous analysis result to generate concrete scheme param
    auto concreteSchemeParam = generateParamByGap<NoiseAnalysis<NoiseModel>>(
        &solver, schemeParam, model);

    LLVM_DEBUG(llvm::dbgs() << "Concrete Scheme Param:\n"
                            << concreteSchemeParam << "\n");

    annotateSchemeParam(getOperation(), concreteSchemeParam, minSlotCount,
                        usePublicKey, encryptionTechniqueExtended);
  }

  void runOnOperation() override {
    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<bgv::SchemeParamAttr>(
                bgv::BGVDialect::kSchemeParamAttrName)) {
      return;
    }

    if (moduleIsOpenfhe(getOperation())) {
      generateFallbackParam(getOperation(), minSlotCount, plaintextModulus,
                            usePublicKey, encryptionTechniqueExtended, 45);
      return;
    }

    // for lattigo, defaults to extended encryption technique
    if (moduleIsLattigo(getOperation())) {
      encryptionTechniqueExtended = true;
    }

    if (model == "bgv-noise-by-bound-coeff-worst-case") {
      bgv::NoiseByBoundCoeffModel model(NoiseModelVariant::WORST_CASE);
      run<bgv::NoiseByBoundCoeffModel>(model);
    } else if (model == "bgv-noise-by-bound-coeff-average-case" ||
               model == "bgv-noise-kpz21") {
      bgv::NoiseByBoundCoeffModel model(NoiseModelVariant::AVERAGE_CASE);
      run<bgv::NoiseByBoundCoeffModel>(model);
    } else if (model == "bgv-noise-by-variance-coeff" ||
               model == "bgv-noise-mp24") {
      bgv::NoiseByVarianceCoeffModel model;
      run<bgv::NoiseByVarianceCoeffModel>(model);
    } else if (model == "bgv-noise-mono") {
      bgv::NoiseCanEmbModel model;
      run<bgv::NoiseCanEmbModel>(model);
    } else {
      emitWarning(getOperation()->getLoc()) << "Unknown noise model.\n";
      generateFallbackParam(getOperation(), minSlotCount, plaintextModulus,
                            usePublicKey, encryptionTechniqueExtended, 45);
    }
  }
};

}  // namespace heir
}  // namespace mlir
