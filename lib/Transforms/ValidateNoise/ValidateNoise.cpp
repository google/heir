#include "lib/Transforms/ValidateNoise/ValidateNoise.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/BFV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BFV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByVarianceCoeffModel.h"
#include "lib/Analysis/NoiseAnalysis/BGV/NoiseCanEmbModel.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

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
      const typename NoiseAnalysis::SchemeParamType &schemeParam,
      const typename NoiseAnalysis::NoiseModel &model) {
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

    auto budget = model.toLogBudget(localParam, noiseState);

    auto boundString = model.toLogBoundString(localParam, noiseState);
    auto budgetString = model.toLogBudgetString(localParam, noiseState);
    auto totalString = model.toLogTotalString(localParam);

    LLVM_DEBUG({
      llvm::dbgs() << "Noise Bound: " << boundString
                   << " Budget: " << budgetString << " Total: " << totalString
                   << " for value: " << value << " " << "\n";
    });

    if (annotateNoiseBound) {
      auto boundStringAttr = StringAttr::get(&getContext(), boundString);
      setAttributeAssociatedWith(value, kArgNoiseBoundAttrName,
                                 boundStringAttr);
    }

    if (budget < 0) {
      return failure();
    }

    return success();
  }

  template <typename NoiseAnalysis>
  LogicalResult validate(
      DataFlowSolver *solver,
      const typename NoiseAnalysis::SchemeParamType &schemeParam,
      const typename NoiseAnalysis::NoiseModel &model) {
    auto res = getOperation()->walk([&](secret::GenericOp genericOp) {
      // check arguments
      for (Value arg : genericOp.getBody()->getArguments()) {
        if (failed(validateNoiseForValue<NoiseAnalysis>(arg, solver,
                                                        schemeParam, model))) {
          return WalkResult::interrupt();
        }
      }

      // check each operation
      // TODO(#1181): handle region bearing ops
      return genericOp.getBody()->walk([&](Operation *op) {
        for (Value result : op->getResults()) {
          if (failed(validateNoiseForValue<NoiseAnalysis>(
                  result, solver, schemeParam, model))) {
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

  template <typename NoiseModel>
  void run(const NoiseModel &model) {
    int maxLevel = getMaxLevel();

    auto schemeParamAttr = getOperation()->getAttrOfType<bgv::SchemeParamAttr>(
        bgv::BGVDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      getOperation()->emitOpError() << "No scheme param found.\n";
      signalPassFailure();
      return;
    }

    // skip validation for Openfhe anyway
    if (moduleIsOpenfhe(getOperation())) {
      return;
    }

    auto schemeParam =
        NoiseModel::SchemeParamType::getSchemeParamFromAttr(schemeParamAttr);
    if (schemeParam.getLevel() < maxLevel) {
      getOperation()->emitOpError()
          << "The level in the scheme param is smaller than the max level.\n";
      signalPassFailure();
      return;
    }

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    // NoiseAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    // NoiseAnalysis depends on SelectVariableNameAnalysis
    SelectVariableNames selectVariableNames(getOperation());
    solver.load<SelectVariableNameAnalysis>(selectVariableNames);

    solver.load<NoiseAnalysis<NoiseModel>>(schemeParam, model);

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    if (failed(
            validate<NoiseAnalysis<NoiseModel>>(&solver, schemeParam, model))) {
      getOperation()->emitOpError() << "Noise validation failed.\n";
      signalPassFailure();
    }
  }

  void runOnOperation() override {
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
    } else if (model == "bfv-noise-by-bound-coeff-worst-case") {
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
    } else {
      getOperation()->emitOpError() << "Unknown noise model.\n";
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
