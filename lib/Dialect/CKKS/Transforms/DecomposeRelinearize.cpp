#include "lib/Dialect/CKKS/Transforms/DecomposeRelinearize.h"

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/Transforms/Patterns.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_DEF_DECOMPOSERELINEARIZE
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

struct DecomposeRelinearize
    : impl::DecomposeRelinearizeBase<DecomposeRelinearize> {
  using DecomposeRelinearizeBase::DecomposeRelinearizeBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DecomposeRelinearizePattern>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
