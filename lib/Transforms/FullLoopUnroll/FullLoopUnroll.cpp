#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h"

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FULLLOOPUNROLL
#include "lib/Transforms/FullLoopUnroll/FullLoopUnroll.h.inc"

struct FullLoopUnroll : impl::FullLoopUnrollBase<FullLoopUnroll> {
  using FullLoopUnrollBase::FullLoopUnrollBase;

  void runOnOperation() override {
    auto walkResult =
        getOperation()->walk<WalkOrder::PostOrder>([&](affine::AffineForOp op) {
          auto result = mlir::affine::loopUnrollFull(op);
          if (failed(result)) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
