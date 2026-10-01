#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Kernel/IR/KernelDialect.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "lib/Transforms/LowerPolynomialEval/Patterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERPOLYNOMIALEVAL
#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h.inc"

static bool hasBackendAttribute(ModuleOp module) {
  if (!module) return false;
  for (NamedAttribute attr : module->getAttrs()) {
    if (!isa<UnitAttr>(attr.getValue())) continue;
    if (attr.getName().strref().starts_with("backend.")) {
      return true;
    }
  }
  return false;
}

struct LowerPolynomialEval
    : impl::LowerPolynomialEvalBase<LowerPolynomialEval> {
  using LowerPolynomialEvalBase::LowerPolynomialEvalBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();

    ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    if (!module) {
      module = getOperation()->getParentOfType<ModuleOp>();
    }

    bool hasKernelChebyshev = false;
    if (module && hasBackendAttribute(module)) {
      auto target = getTargetConfig(module);
      if (succeeded(target)) {
        hasKernelChebyshev = target->has_kernel_chebyshev;
      }
    }

    RewritePatternSet patterns(context);

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run SecretnessAnalysis.\n";
      return signalPassFailure();
    }

    switch (method) {
      case PolynomialApproximationMethod::Automatic:
        patterns.add<LowerViaHorner, LowerViaPatersonStockmeyerMonomial>(
            context, /*force=*/false);
        if (hasKernelChebyshev) {
          patterns.add<LowerToKernelEvalChebyshev>(context, solver,
                                                   /*force=*/false);
          patterns.add<LowerViaPatersonStockmeyerChebyshev>(
              context,
              /*force=*/false, minCoefficientThreshold);
        } else {
          patterns.add<LowerViaPatersonStockmeyerChebyshev>(
              context,
              /*force=*/false, minCoefficientThreshold);
        }
        break;
      case PolynomialApproximationMethod::Horner:
        patterns.add<LowerViaHorner>(context, /*force=*/true);
        break;
      case PolynomialApproximationMethod::PatersonStockmeyer:
        patterns.add<LowerViaPatersonStockmeyerMonomial>(context,
                                                         /*force=*/true);
        break;
      case PolynomialApproximationMethod::PatersonStockmeyerChebyshev:
        if (hasKernelChebyshev) {
          patterns.add<LowerToKernelEvalChebyshev>(context, solver,
                                                   /*force=*/true);
          patterns.add<LowerViaPatersonStockmeyerChebyshev>(
              context,
              /*force=*/true, minCoefficientThreshold);
        } else {
          patterns.add<LowerViaPatersonStockmeyerChebyshev>(
              context,
              /*force=*/true, minCoefficientThreshold);
        }
        break;
      default:
        getOperation()->emitError() << "Unknown lowering method: " << method;
        signalPassFailure();
        return;
    }

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
