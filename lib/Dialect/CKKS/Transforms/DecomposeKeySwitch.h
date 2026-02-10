#ifndef LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_KEYSWITCH_H_
#define LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_KEYSWITCH_H_

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_DECL_DECOMPOSEKEYSWITCH
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

struct DecomposeKeySwitchPattern : public OpRewritePattern<KeySwitchInnerOp> {
  using OpRewritePattern<KeySwitchInnerOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(KeySwitchInnerOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_KEYSWITCH_H_
