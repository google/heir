#ifndef LIB_DIALECT_CKKS_TRANSFORMS_PATTERNS_H_
#define LIB_DIALECT_CKKS_TRANSFORMS_PATTERNS_H_

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

struct DecomposeRelinearizePattern : public OpRewritePattern<RelinearizeOp> {
  using OpRewritePattern<RelinearizeOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(RelinearizeOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
#endif  // LIB_DIALECT_CKKS_TRANSFORMS_PATTERNS_H_
