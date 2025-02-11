#include "lib/Transforms/ValidateNoise/ValidateNoise.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
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
    } else {
      getOperation()->emitOpError() << "Unknown noise model.\n";
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
