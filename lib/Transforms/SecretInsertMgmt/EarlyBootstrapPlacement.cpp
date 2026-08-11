#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"       // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"           // from @llvm-project
// IWYU pragma: end_keep

#include "lib/Dialect/ModuleAttributes.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_EARLYBOOTSTRAPPLACEMENT
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct EarlyBootstrapPlacement
    : impl::EarlyBootstrapPlacementBase<EarlyBootstrapPlacement> {
  using EarlyBootstrapPlacementBase::EarlyBootstrapPlacementBase;

  void runOnOperation() override {
    if (!moduleIsCKKS(getOperation())) {
      return;
    }
    int budget = levelBudget;
    int absoluteWaterline = budget - bootstrapWaterline;
    int idCounter = 0;
    insertBootstrapWaterLine(getOperation(), absoluteWaterline, budget,
                             /*bootstrapLevelsConsumed=*/0,
                             /*includeFloats=*/true, &idCounter,
                             /*onlyHoist=*/false);
  }
};

}  // namespace heir
}  // namespace mlir
