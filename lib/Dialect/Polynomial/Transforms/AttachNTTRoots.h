#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_ATTACHNTTROOTS_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_ATTACHNTTROOTS_H_

#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL_ATTACHNTTROOTS
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

struct RootCache {
  // Limb-level caching
  DenseMap<RingAttr, Attribute> nttRoots;
  DenseMap<RingAttr, Attribute> inttRoots;
};

struct AttachNTTRootsPattern : public OpRewritePattern<NTTOp> {
  using OpRewritePattern<NTTOp>::OpRewritePattern;
  AttachNTTRootsPattern(MLIRContext* context, RootCache& cache)
      : OpRewritePattern<NTTOp>(context), cache(cache) {}

  LogicalResult matchAndRewrite(NTTOp op,
                                PatternRewriter& rewriter) const override;

 private:
  RootCache& cache;
};

struct AttachINTTRootsPattern : public OpRewritePattern<INTTOp> {
  using OpRewritePattern<INTTOp>::OpRewritePattern;
  AttachINTTRootsPattern(MLIRContext* context, RootCache& cache)
      : OpRewritePattern<INTTOp>(context), cache(cache) {}

  LogicalResult matchAndRewrite(INTTOp op,
                                PatternRewriter& rewriter) const override;

 private:
  RootCache& cache;
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_ATTACHNTTROOTS_H_
