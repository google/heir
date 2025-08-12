#include "lib/Transforms/LowerUnpack/LowerUnpack.h"

#include <utility>

#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// iwyu pragma: begin_keep
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project
// iwyu pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERUNPACK
#include "lib/Transforms/LowerUnpack/LowerUnpack.h.inc"

struct LowerUnpack : impl::LowerUnpackBase<LowerUnpack> {
  using LowerUnpackBase::LowerUnpackBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerUnpackOp>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
