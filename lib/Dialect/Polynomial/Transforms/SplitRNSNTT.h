#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_SPLITRNSNTT_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_SPLITRNSNTT_H_

#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL_SPLITRNSNTT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

struct SplitRNSNTTPattern : public OpRewritePattern<NTTOp> {
  using OpRewritePattern<NTTOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(NTTOp op,
                                PatternRewriter& rewriter) const override;
};

struct SplitRNSINTTPattern : public OpRewritePattern<INTTOp> {
  using OpRewritePattern<INTTOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(INTTOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_SPLITRNSNTT_H_
