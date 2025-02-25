#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BFV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#define DEBUG_TYPE "GenerateParamBFV"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMBFV
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

struct GenerateParamBFV : impl::GenerateParamBFVBase<GenerateParamBFV> {
  using GenerateParamBFVBase::GenerateParamBFVBase;

  // assume only one main func
  // also assume max level at entry
  // also assume first genericOp arg is secret
  int getMaxLevel() {
    int maxLevel = 0;
    getOperation()->walk([&](func::FuncOp funcOp) {
      funcOp->walk([&](secret::GenericOp genericOp) {
        if (genericOp.getBody()->getNumArguments() > 0) {
          maxLevel = getLevelFromMgmtAttr(genericOp.getBody()->getArgument(0));
        }
      });
    });
    return maxLevel;
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
            schemeParam.getPlaintextModulus()));
  }

  template <typename NoiseAnalysis>
  typename NoiseAnalysis::SchemeParamType generateParamByMaxNoise(
      DataFlowSolver *solver,
      const typename NoiseAnalysis::SchemeParamType &schemeParam) {
    using NoiseModel = typename NoiseAnalysis::NoiseModel;
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
      return NoiseModel::toLogBound(localParam, noiseLattice->getValue());
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
            qiSize, schemeParam.getPlaintextModulus(), slotNumber);

    return concreteSchemeParam;
  }

  template <typename NoiseAnalysis>
  void run() {
    int maxLevel = getMaxLevel();

    // plaintext modulus from command line option
    auto schemeParam =
        NoiseAnalysis::SchemeParamType::getConservativeSchemeParam(
            maxLevel, plaintextModulus, slotNumber);

    LLVM_DEBUG(llvm::dbgs() << "Conservative Scheme Param:\n"
                            << schemeParam << "\n");

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    // NoiseAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    solver.load<NoiseAnalysis>(schemeParam);
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    // use previous analysis result to generate concrete scheme param
    auto concreteSchemeParam =
        generateParamByMaxNoise<NoiseAnalysis>(&solver, schemeParam);

    LLVM_DEBUG(llvm::dbgs() << "Concrete Scheme Param:\n"
                            << concreteSchemeParam << "\n");

    annotateSchemeParam(concreteSchemeParam);
  }

  void generateFallbackParam() {
    // generate fallback scheme parameters
    auto maxLevel = getMaxLevel();
    std::vector<double> logPrimes(maxLevel + 1,
                                  modBits);  // all primes of modBits bits

    auto schemeParam = bgv::SchemeParam::getConcreteSchemeParam(
        logPrimes, plaintextModulus, slotNumber);

    annotateSchemeParam(schemeParam);
  }

  void runOnOperation() override {
    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<bgv::SchemeParamAttr>(
                bgv::BGVDialect::kSchemeParamAttrName)) {
      getOperation()->emitRemark()
          << "Scheme parameters already exist. Skipping generation.\n";
      return;
    }

    if (model == "bfv-noise-by-bound-coeff-worst-case-pk") {
      run<NoiseAnalysis<bfv::NoiseByBoundCoeffWorstCasePkModel>>();
    } else if (model == "bfv-noise-by-bound-coeff-average-case-pk") {
      run<NoiseAnalysis<bfv::NoiseByBoundCoeffAverageCasePkModel>>();
    } else if (model == "bfv-noise-by-bound-coeff-worst-case-sk") {
      run<NoiseAnalysis<bfv::NoiseByBoundCoeffWorstCaseSkModel>>();
    } else if (model == "bfv-noise-by-bound-coeff-average-case-sk") {
      run<NoiseAnalysis<bfv::NoiseByBoundCoeffAverageCaseSkModel>>();
    } else {
      getOperation()->emitWarning() << "Unknown noise model.\n";
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
