#include "lib/Transforms/ReluCanonicalizations/ReluCanonicalizations.h"

#include <utility>

#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RELUCANONICALIZATIONS
#include "lib/Transforms/ReluCanonicalizations/ReluCanonicalizations.h.inc"

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Transforms/ReluCanonicalizations/Rewrites.cpp.inc"

struct ReluCanonicalizations
    : impl::ReLUCanonicalizationsBase<ReluCanonicalizations> {
  using ReLUCanonicalizationsBase::ReLUCanonicalizationsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    populateWithGenerated(patterns);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
