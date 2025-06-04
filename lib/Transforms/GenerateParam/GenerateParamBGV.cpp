#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <vector>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseCanEmbModel.h"
#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVEnums.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#define DEBUG_TYPE "GenerateParamBGV"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMBGV
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

struct GenerateParamBGV : impl::GenerateParamBGVBase<GenerateParamBGV> {
  using GenerateParamBGVBase::GenerateParamBGVBase;

  template <typename NoiseAnalysis>
  typename NoiseAnalysis::SchemeParamType generateParamByGap(
      DataFlowSolver *solver,
      const typename NoiseAnalysis::SchemeParamType &schemeParam,
      const typename NoiseAnalysis::NoiseModel &noiseModel) {
    using NoiseLatticeType = typename NoiseAnalysis::LatticeType;
    using LocalParamType = typename NoiseAnalysis::LocalParamType;

    // for level i, the biggest gap observed.
    std::map<int, double> levelToGap;

    auto updateLevelToGap = [&](int level, double gap) {
      if (levelToGap.count(level) == 0) {
        levelToGap[level] = gap;
      } else {
        levelToGap[level] = std::max(levelToGap.at(level), gap);
      }
    };

    auto getLocalParam = [&](Value value) {
      auto level = getLevelFromMgmtAttr(value);
      auto dimension = getDimensionFromMgmtAttr(value);
      return LocalParamType(&schemeParam, level, dimension);
    };

    auto getBound = [&](Value value) {
      auto localParam = getLocalParam(value);
      auto noiseLattice = solver->lookupState<NoiseLatticeType>(value);
      return noiseModel.toLogBound(localParam, noiseLattice->getValue());
    };

    auto firstModSize = 0;

    getOperation()->walk([&](secret::GenericOp genericOp) {
      // gaps caused by mod reduce
      genericOp.getBody()->walk([&](mgmt::ModReduceOp op) {
        auto operandBound = getBound(op.getOperand());
        auto resultBound = getBound(op.getResult());
        // the gap between the operand and result
        updateLevelToGap(getLevelFromMgmtAttr(op.getOperand()),
                         operandBound - resultBound);
        return WalkResult::advance();
      });

      // find the max noise for the first level
      genericOp.getBody()->walk([&](Operation *op) {
        for (Value result : op->getResults()) {
          if (getLevelFromMgmtAttr(result) == 0) {
            auto bound = getBound(result);
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

    for (auto &[level, gap] : levelToGap) {
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
            qiSize, schemeParam.getPlaintextModulus(), slotNumber, usePublicKey,
            encryptionTechniqueExtended);

    return concreteSchemeParam;
  }

  void annotateSchemeParam(const bgv::SchemeParam &schemeParam) {
    getOperation()->setAttr(
        bgv::BGVDialect::kSchemeParamAttrName,
        bgv::SchemeParamAttr::get(
            &getContext(), log2(schemeParam.getRingDim()),

            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(schemeParam.getQi())),
            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(schemeParam.getPi())),
            schemeParam.getPlaintextModulus(),
            usePublicKey ? bgv::BGVEncryptionType::pk
                         : bgv::BGVEncryptionType::sk,
            encryptionTechniqueExtended
                ? bgv::BGVEncryptionTechnique::extended
                : bgv::BGVEncryptionTechnique::standard));
  }

  template <typename NoiseModel>
  void run(const NoiseModel &model) {
    std::optional<int> maxLevel = getMaxLevel(getOperation());

    // plaintext modulus from command line option
    auto schemeParam = NoiseModel::SchemeParamType::getConservativeSchemeParam(
        maxLevel.value_or(0), plaintextModulus, slotNumber, usePublicKey,
        encryptionTechniqueExtended);

    LLVM_DEBUG(llvm::dbgs() << "Conservative Scheme Param:\n"
                            << schemeParam << "\n");

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    // NoiseAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    solver.load<NoiseAnalysis<NoiseModel>>(schemeParam, model);
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    // use previous analysis result to generate concrete scheme param
    auto concreteSchemeParam = generateParamByGap<NoiseAnalysis<NoiseModel>>(
        &solver, schemeParam, model);

    LLVM_DEBUG(llvm::dbgs() << "Concrete Scheme Param:\n"
                            << concreteSchemeParam << "\n");

    annotateSchemeParam(concreteSchemeParam);
  }

  void generateFallbackParam() {
    // generate fallback scheme parameters
    std::optional<int> maxLevel = getMaxLevel(getOperation());
    std::vector<double> logPrimes(maxLevel.value_or(0) + 1,
                                  45);  // all primes of 45 bits

    auto schemeParam = bgv::SchemeParam::getConcreteSchemeParam(
        logPrimes, plaintextModulus, slotNumber, usePublicKey,
        encryptionTechniqueExtended);

    annotateSchemeParam(schemeParam);
  }

  void runOnOperation() override {
    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<bgv::SchemeParamAttr>(
                bgv::BGVDialect::kSchemeParamAttrName)) {
      return;
    }

    if (moduleIsOpenfhe(getOperation())) {
      generateFallbackParam();
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
      getOperation()->emitWarning() << "Unknown noise model.\n";
      generateFallbackParam();
    }
  }
};

}  // namespace heir
}  // namespace mlir
