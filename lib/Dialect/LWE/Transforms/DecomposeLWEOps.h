#ifndef LIB_DIALECT_LWE_TRANSFORMS_DECOMPOSE_LWE_OPS_H_
#define LIB_DIALECT_LWE_TRANSFORMS_DECOMPOSE_LWE_OPS_H_

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DECL_DECOMPOSELWEOPS
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

struct DecomposeKeySwitchPattern : public OpRewritePattern<KeySwitchInnerOp> {
  using OpRewritePattern<KeySwitchInnerOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(KeySwitchInnerOp op,
                                PatternRewriter& rewriter) const override;
};

struct DecomposeModDownPattern : public OpRewritePattern<ModDownOp> {
  using OpRewritePattern<ModDownOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(ModDownOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_TRANSFORMS_DECOMPOSE_LWE_OPS_H_
