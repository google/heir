#include "lib/Dialect/Cheddar/Transforms/FuseOps.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

#define GEN_PASS_DEF_CHEDDARFUSEOPS
#include "lib/Dialect/Cheddar/Transforms/FuseOps.h.inc"

namespace {

bool isFollowedBySingleUseRescale(ValueRange results) {
  if (results.size() != 1 || !results.front().hasOneUse()) return false;
  Value value = results.front();
  auto rescaleOp = dyn_cast<RescaleOp>(*value.getUsers().begin());
  if (!rescaleOp) return false;
  auto relinOp = value.getDefiningOp<RelinearizeOp>();
  return relinOp && rescaleOp.getCtx() == relinOp.getCtx();
}

IntegerAttr getConstantDistance(OpBuilder& builder, Value dynamicDistance,
                                IntegerAttr staticDistance) {
  if (staticDistance) return staticDistance;
  if (dynamicDistance) {
    if (std::optional<int64_t> constantDistance =
            getConstantIntValue(dynamicDistance)) {
      return builder.getI64IntegerAttr(*constantDistance);
    }
  }
  return nullptr;
}

#include "lib/Dialect/Cheddar/Transforms/FuseOpsPatterns.cpp.inc"

struct CheddarFuseOps : public impl::CheddarFuseOpsBase<CheddarFuseOps> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    populateWithGenerated(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
