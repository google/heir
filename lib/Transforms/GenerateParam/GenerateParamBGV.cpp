#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseCanEmbModel.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
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

#define DEBUG_TYPE "GenerateParamBGV"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMBGV
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

struct GenerateParamBGV : impl::GenerateParamBGVBase<GenerateParamBGV> {
  using GenerateParamBGVBase::GenerateParamBGVBase;

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

  template <typename NoiseAnalysis>
  typename NoiseAnalysis::SchemeParamType generateParamByGap(
      DataFlowSolver *solver,
      const typename NoiseAnalysis::SchemeParamType &schemeParam) {
    using NoiseModel = typename NoiseAnalysis::NoiseModel;
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
      return NoiseModel::toLogBound(localParam, noiseLattice->getValue());
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

  template <typename NoiseAnalysis>
  void run() {
    int maxLevel = getMaxLevel();

    // plaintext modulus from command line option
    auto schemeParam =
        NoiseAnalysis::SchemeParamType::getConservativeSchemeParam(
            maxLevel, plaintextModulus, slotNumber, usePublicKey,
            encryptionTechniqueExtended);

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
        generateParamByGap<NoiseAnalysis>(&solver, schemeParam);

    LLVM_DEBUG(llvm::dbgs() << "Concrete Scheme Param:\n"
                            << concreteSchemeParam << "\n");

    annotateSchemeParam(concreteSchemeParam);
  }

  void generateFallbackParam() {
    // generate fallback scheme parameters
    auto maxLevel = getMaxLevel();
    std::vector<double> logPrimes(maxLevel + 1, 45);  // all primes of 45 bits

    auto schemeParam = bgv::SchemeParam::getConcreteSchemeParam(
        logPrimes, plaintextModulus, slotNumber, usePublicKey,
        encryptionTechniqueExtended);

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

    if (model == "bgv-noise-by-bound-coeff-worst-case") {
      run<NoiseAnalysis<bgv::NoiseByBoundCoeffWorstCaseModel>>();
    } else if (model == "bgv-noise-by-bound-coeff-average-case" ||
               model == "bgv-noise-kpz21") {
      run<NoiseAnalysis<bgv::NoiseByBoundCoeffAverageCaseModel>>();
    } else if (model == "bgv-noise-by-variance-coeff" ||
               model == "bgv-noise-mp24") {
      run<NoiseAnalysis<bgv::NoiseByVarianceCoeffModel>>();
    } else if (model == "bgv-noise-mono") {
      run<NoiseAnalysis<bgv::NoiseCanEmbModel>>();
    } else {
      getOperation()->emitWarning() << "Unknown noise model.\n";
      generateFallbackParam();
    }
  }
};

}  // namespace heir
}  // namespace mlir
