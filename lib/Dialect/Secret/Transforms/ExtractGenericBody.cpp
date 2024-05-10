#include "lib/Dialect/Secret/Transforms/ExtractGenericBody.h"

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETEXTRACTGENERICBODY
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

struct ExtractGenericBodyPass
    : impl::SecretExtractGenericBodyBase<ExtractGenericBodyPass> {
  using SecretExtractGenericBodyBase::SecretExtractGenericBodyBase;

  void runOnOperation() override {
    mlir::IRRewriter builder(&getContext());

    auto result = getOperation()->walk([&](secret::GenericOp op) {
      if (failed(extractGenericBody(op, builder))) {
        op.emitError() << "Failed to extract generic body";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
