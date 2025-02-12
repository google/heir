#include "lib/Transforms/ValidateNoise/ValidateNoise.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
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

#define DEBUG_TYPE "ValidateNoise"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_VALIDATENOISE
#include "lib/Transforms/ValidateNoise/ValidateNoise.h.inc"

struct ValidateNoise : impl::ValidateNoiseBase<ValidateNoise> {
  using ValidateNoiseBase::ValidateNoiseBase;

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
  LogicalResult validateNoiseForValue(
      Value value, DataFlowSolver *solver,
      const typename NoiseAnalysis::SchemeParamType &schemeParam) {
    using NoiseModel = typename NoiseAnalysis::NoiseModel;
    using NoiseLatticeType = typename NoiseAnalysis::LatticeType;
    using LocalParamType = typename NoiseAnalysis::LocalParamType;

    auto getLocalParam = [&](Value value) {
      auto level = getLevelFromMgmtAttr(value);
      auto dimension = getDimensionFromMgmtAttr(value);
      return LocalParamType(&schemeParam, level, dimension);
    };

    auto secretness = isSecret(value, solver);
    if (!secretness) {
      return success();
    }

    const auto *noiseLattice = solver->lookupState<NoiseLatticeType>(value);
    if (!noiseLattice || !noiseLattice->getValue().isInitialized()) {
      return failure();
    }

    auto noiseState = noiseLattice->getValue();
    auto localParam = getLocalParam(value);

    auto budget = NoiseModel::toLogBudget(localParam, noiseState);

    LLVM_DEBUG({
      auto boundString = NoiseModel::toLogBoundString(localParam, noiseState);
      auto budgetString = NoiseModel::toLogBudgetString(localParam, noiseState);
      auto totalString = NoiseModel::toLogTotalString(localParam);
      llvm::dbgs() << "Noise Bound: " << boundString
                   << " Budget: " << budgetString << " Total: " << totalString
                   << " for value: " << value << " " << "\n";
    });

    if (budget < 0) {
      return failure();
    }

    return success();
  }

  template <typename NoiseAnalysis>
  LogicalResult validate(
      DataFlowSolver *solver,
      const typename NoiseAnalysis::SchemeParamType &schemeParam) {
    auto res = getOperation()->walk([&](secret::GenericOp genericOp) {
      // check arguments
      for (Value arg : genericOp.getBody()->getArguments()) {
        if (failed(validateNoiseForValue<NoiseAnalysis>(arg, solver,
                                                        schemeParam))) {
          return WalkResult::interrupt();
        }
      }

      // check each operation
      // TODO(#1181): handle region bearing ops
      return genericOp.getBody()->walk([&](Operation *op) {
        for (Value result : op->getResults()) {
          if (failed(validateNoiseForValue<NoiseAnalysis>(result, solver,
                                                          schemeParam))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    });
    if (res == WalkResult::interrupt()) {
      return failure();
    }
    return success();
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
            firstModSize = std::max(firstModSize, 1 + int(ceil(bound)));
          }
        }
        return WalkResult::advance();
      });
    });

    auto maxLevel = levelToGap.size() + 1;
    auto qiSize = std::vector<double>(maxLevel, 0);
    qiSize[0] = firstModSize;

    for (auto &[level, gap] : levelToGap) {
      qiSize[level] = int(ceil(gap));
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
            schemeParam.getPlaintextModulus(), qiSize);

    LLVM_DEBUG(llvm::dbgs() << "Concrete Scheme Param:\n"
                            << concreteSchemeParam << "\n");

    return concreteSchemeParam;
  }

  template <typename NoiseAnalysis>
  void run() {
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    // NoiseAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();

    int maxLevel = getMaxLevel();

    // plaintext modulus from command line option
    auto schemeParam =
        NoiseAnalysis::SchemeParamType::getConservativeSchemeParam(
            maxLevel, plaintextModulus);

    LLVM_DEBUG(llvm::dbgs() << "Conservative Scheme Param:\n"
                            << schemeParam << "\n");

    solver.load<NoiseAnalysis>(schemeParam);

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    if (failed(validate<NoiseAnalysis>(&solver, schemeParam))) {
      getOperation()->emitOpError() << "Noise validation failed.\n";
      signalPassFailure();
      return;
    }

    auto concreteSchemeParam =
        generateParamByGap<NoiseAnalysis>(&solver, schemeParam);

    if (failed(validate<NoiseAnalysis>(&solver, concreteSchemeParam))) {
      getOperation()->emitOpError()
          << "Noise validation failed for generated param.\n";
      signalPassFailure();
      return;
    }

    // annotate scheme param
    getOperation()->setAttr(
        bgv::BGVDialect::kSchemeParamAttrName,
        bgv::SchemeParamAttr::get(
            &getContext(), log2(concreteSchemeParam.getRingDim()),

            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(concreteSchemeParam.getQi())),
            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(concreteSchemeParam.getPi())),
            concreteSchemeParam.getPlaintextModulus()));
  }

  void runOnOperation() override {
    if (model == "bgv-noise-by-bound-coeff-worst-case-pk") {
      run<NoiseAnalysis<bgv::NoiseByBoundCoeffWorstCasePkModel>>();
    } else if (model == "bgv-noise-by-bound-coeff-average-case-pk") {
      run<NoiseAnalysis<bgv::NoiseByBoundCoeffAverageCasePkModel>>();
    } else if (model == "bgv-noise-by-bound-coeff-worst-case-sk") {
      run<NoiseAnalysis<bgv::NoiseByBoundCoeffWorstCaseSkModel>>();
    } else if (model == "bgv-noise-by-bound-coeff-average-case-sk") {
      run<NoiseAnalysis<bgv::NoiseByBoundCoeffAverageCaseSkModel>>();
    } else if (model == "bgv-noise-by-variance-coeff-pk") {
      run<NoiseAnalysis<bgv::NoiseByVarianceCoeffPkModel>>();
    } else if (model == "bgv-noise-by-variance-coeff-sk") {
      run<NoiseAnalysis<bgv::NoiseByVarianceCoeffSkModel>>();
    } else {
      getOperation()->emitOpError() << "Unknown noise model.\n";
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
