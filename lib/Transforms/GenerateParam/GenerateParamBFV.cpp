#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BFV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BFV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BFV/NoiseCanEmbModel.h"
#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVEnums.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#define DEBUG_TYPE "GenerateParamBFV"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMBFV
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

struct GenerateParamBFV : impl::GenerateParamBFVBase<GenerateParamBFV> {
  using GenerateParamBFVBase::GenerateParamBFVBase;

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

  template <typename NoiseAnalysis>
  typename NoiseAnalysis::SchemeParamType generateParamByMaxNoise(
      DataFlowSolver *solver,
      const typename NoiseAnalysis::SchemeParamType &schemeParam,
      const typename NoiseAnalysis::NoiseModel &noiseModel) {
    using NoiseLatticeType = typename NoiseAnalysis::LatticeType;
    using LocalParamType = typename NoiseAnalysis::LocalParamType;

    double maxNoiseBound = 0.0;

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

    getOperation()->walk([&](secret::GenericOp genericOp) {
      // find the max noise
      genericOp.getBody()->walk([&](Operation *op) {
        for (Value result : op->getResults()) {
          auto bound = getBound(result);
          maxNoiseBound = std::max(maxNoiseBound, bound);
        }
        return WalkResult::advance();
      });
    });

    // logQ >= log||e|| + log(t) + 1
    auto logQ =
        1 + int(ceil(maxNoiseBound + log2(schemeParam.getPlaintextModulus())));

    LLVM_DEBUG(llvm::dbgs()
               << "Max Noise Bound: " << static_cast<int>(maxNoiseBound)
               << " logQ: " << logQ << "\n");

    int numqi = ceil(static_cast<double>(logQ) / modBits);
    auto qiSize = std::vector<double>(numqi, 0);
    for (auto i = 0; i < numqi; ++i) {
      qiSize[i] = std::min<int>(modBits, logQ);
      logQ -= modBits;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "logqi: ";
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
    dataflow::loadBaselineAnalyses(solver);
    // NoiseAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    solver.load<NoiseAnalysis<NoiseModel>>(schemeParam, model);
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    // use previous analysis result to generate concrete scheme param
    auto concreteSchemeParam =
        generateParamByMaxNoise<NoiseAnalysis<NoiseModel>>(&solver, schemeParam,
                                                           model);

    LLVM_DEBUG(llvm::dbgs() << "Concrete Scheme Param:\n"
                            << concreteSchemeParam << "\n");

    annotateSchemeParam(concreteSchemeParam);
  }

  void generateFallbackParam() {
    // generate fallback scheme parameters
    auto maxLevel = getMaxLevel(getOperation());
    std::vector<double> logPrimes(maxLevel.value_or(0) + 1,
                                  modBits);  // all primes of modBits bits

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
      // no need to re-set level below as fallback parameter has
      // the same level as the max level
      return;
    }

    // for lattigo, defaults to extended encryption technique
    if (moduleIsLattigo(getOperation())) {
      encryptionTechniqueExtended = true;
    }

    if (model == "bfv-noise-by-bound-coeff-worst-case") {
      bfv::NoiseByBoundCoeffModel model(NoiseModelVariant::WORST_CASE);
      run<bfv::NoiseByBoundCoeffModel>(model);
    } else if (model == "bfv-noise-by-bound-coeff-average-case" ||
               model == "bfv-noise-kpz21") {
      bfv::NoiseByBoundCoeffModel model(NoiseModelVariant::AVERAGE_CASE);
      run<bfv::NoiseByBoundCoeffModel>(model);
    } else if (model == "bfv-noise-by-variance-coeff" ||
               model == "bfv-noise-bmcm23") {
      bfv::NoiseByVarianceCoeffModel model;
      run<bfv::NoiseByVarianceCoeffModel>(model);
    } else if (model == "bfv-noise-canon-emb") {
      bfv::NoiseCanEmbModel model;
      run<bfv::NoiseCanEmbModel>(model);
    } else {
      emitWarning(getOperation()->getLoc()) << "Unknown noise model.\n";
      generateFallbackParam();
    }

    auto schemeParamAttr = getOperation()->getAttrOfType<bgv::SchemeParamAttr>(
        bgv::BGVDialect::kSchemeParamAttrName);

    // annotate mgmt attribute with all levels set to the generated parameter.
    // note that the parameter generation process may produce 'level'
    // less than the mulDepth as we might not need that much.
    auto level = schemeParamAttr.getQ().getSize() - 1;
    OpPassManager annotateMgmtPipeline("builtin.module");
    mgmt::AnnotateMgmtOptions annotateMgmtOptions;
    annotateMgmtOptions.baseLevel = level;
    annotateMgmtPipeline.addPass(mgmt::createAnnotateMgmt(annotateMgmtOptions));
    (void)runPipeline(annotateMgmtPipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
