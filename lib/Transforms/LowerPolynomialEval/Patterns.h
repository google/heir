#ifndef LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_PATTERNS_H_
#define LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_PATTERNS_H_

#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

// Lowering patterns for polynomial.eval.

namespace mlir {
namespace heir {

struct LoweringBase : public OpRewritePattern<polynomial::EvalOp> {
  LoweringBase(mlir::MLIRContext* context, bool force = false)
      : mlir::OpRewritePattern<polynomial::EvalOp>(context), force(force) {}

  bool shouldForce() const { return force; }

 private:
  // Force the use of this pattern, ignoring any heuristics on whether to apply
  // it.
  const bool force;
};

// Lower polynomial.eval that uses a monomial float polynomial to a series of
// adds and muls via Horner's method. Supports scalar and tensor operands of
// floating point types.
struct LowerViaHorner : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                PatternRewriter& rewriter) const override;
};

// Lower polynomial.eval that uses a monomial float polynomial to a series of
// adds and muls via the Paterson-Stockmeyer method. Supports scalar and tensor
// operands of floating point types.
struct LowerViaPatersonStockmeyerMonomial : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                PatternRewriter& rewriter) const override;
};

// Lower polynomial.eval that uses a Chebyshev float polynomial to a series of
// adds and muls via the Paterson-Stockmeyer method. Supports scalar and tensor
// operands of floating point types.
struct LowerViaPatersonStockmeyerChebyshev : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_PATTERNS_H_
