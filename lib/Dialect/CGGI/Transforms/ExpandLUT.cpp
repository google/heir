#include "lib/Dialect/CGGI/Transforms/ExpandLUT.h"

#include <cassert>
#include <utility>

#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "expand-lut"

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DEF_EXPANDLUT
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

namespace alignment {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "lib/Dialect/CGGI/Transforms/ExpandLUT.cpp.inc"
}  // namespace alignment

struct ExpandLutLinComb : public OpRewritePattern<LutLinCombOp> {
  ExpandLutLinComb(mlir::MLIRContext *context)
      : OpRewritePattern<LutLinCombOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(LutLinCombOp op,
                                PatternRewriter &rewriter) const override {
    Type scalarTy = rewriter.getIntegerType(widthFromEncodingAttr(
        cast<lwe::LWECiphertextType>(op.getInputs().front().getType())
            .getEncoding()));
    // Use LWE operations to create the linear combination of inputs.
    Value result;
    for (auto [input, coeff] :
         llvm::zip(op.getInputs(), op.getCoefficients())) {
      auto scaled = rewriter.create<lwe::MulScalarOp>(
          op.getLoc(), input,
          rewriter.create<arith::ConstantOp>(
              op.getLoc(), rewriter.getIntegerAttr(scalarTy, coeff)));
      result = result ? rewriter.create<lwe::AddOp>(op.getLoc(), result, scaled)
                            .getResult()
                      : scaled.getResult();
    }
    rewriter.replaceOpWithNewOp<ProgrammableBootstrapOp>(op, result,
                                                         op.getLookupTable());
    return success();
  }
};

struct ExpandLUT : impl::ExpandLUTBase<ExpandLUT> {
  using ExpandLUTBase::ExpandLUTBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    // Add patterns generated from DRR
    alignment::populateWithGenerated(patterns);
    patterns.add<ExpandLutLinComb>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
