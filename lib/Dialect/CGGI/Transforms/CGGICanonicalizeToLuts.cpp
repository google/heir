#include "lib/Dialect/CGGI/Transforms/CGGICanonicalizeToLuts.h"

#include <iostream>

#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DEF_CGGICANONICALIZETOLUTS
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

template <class SourceLut>
struct RewriteLutToLincomb : public OpRewritePattern<SourceLut> {
  RewriteLutToLincomb(MLIRContext *context)
      : OpRewritePattern<SourceLut>(context) {}
  LogicalResult matchAndRewrite(SourceLut op,
                                PatternRewriter &rewriter) const override {
    if constexpr (std::is_same<SourceLut, cggi::Lut3Op>()) {
      rewriter.replaceOpWithNewOp<LutLinCombOp>(
          op, op->getOperands(), rewriter.getDenseI32ArrayAttr({1, 2, 4}),
          op.getLookupTable());
      return success();
    } else if constexpr (std::is_same<SourceLut, cggi::Lut2Op>()) {
      rewriter.replaceOpWithNewOp<LutLinCombOp>(
          op, op->getOperands(), rewriter.getDenseI32ArrayAttr({1, 2}),
          op.getLookupTable());
      return success();
    }
    return failure();
  }
};

template <class SourceOp, int binaryLut, int ternaryLut>
struct RewriteCGGIOp : public OpRewritePattern<SourceOp> {
  RewriteCGGIOp(MLIRContext *context) : OpRewritePattern<SourceOp>(context) {}
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Value> operands(op->getOperands());

    if (operands.size() != 2 && operands.size() != 3) return failure();
    if (operands.size() == 2) {
      rewriter.replaceOpWithNewOp<cggi::LutLinCombOp>(op, operands);
      op->setAttr("coefficients", rewriter.getDenseI32ArrayAttr({1, 2}));
      op->setAttr("lookup_table", rewriter.getIntegerAttr(
                                      rewriter.getIntegerType(4), binaryLut));

      return success();
    }
    rewriter.replaceOpWithNewOp<cggi::LutLinCombOp>(op, operands);
    op->setAttr("coefficients", rewriter.getDenseI32ArrayAttr({1, 2, 4}));
    op->setAttr("lookup_table", rewriter.getIntegerAttr(
                                    rewriter.getIntegerType(4), ternaryLut));

    return success();
  }
};

using RewriteCGGIAnd = RewriteCGGIOp<cggi::AndOp, 8, 128>;
using RewriteCGGIOr = RewriteCGGIOp<cggi::OrOp, 14, 254>;
using RewriteCGGIXor = RewriteCGGIOp<cggi::XorOp, 6, 150>;
// 01101001
// 2^0 + 2^3 + 2^5 + 2^6 = 1 + 8 + 32 + 64 = 9 + 32 + 64 = 41 + 64 = 105??
// 000, 011, 101, 110
using RewriteCGGIXNor = RewriteCGGIOp<cggi::XNorOp, 9, 105>;
// 10, 01, 00
using RewriteCGGINAnd = RewriteCGGIOp<cggi::NandOp, 7, 127>;

struct CGGICanonicalizeToLuts
    : impl::CGGICanonicalizeToLutsBase<CGGICanonicalizeToLuts> {
  using CGGICanonicalizeToLutsBase::CGGICanonicalizeToLutsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<RewriteCGGIAnd, RewriteCGGIOr, RewriteCGGIXor, RewriteCGGIXNor,
                 RewriteCGGINAnd, RewriteLutToLincomb<cggi::Lut2Op>,
                 RewriteLutToLincomb<cggi::Lut3Op>>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace cggi
}  // namespace heir
}  // namespace mlir