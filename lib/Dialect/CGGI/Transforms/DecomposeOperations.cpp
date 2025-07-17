#include "lib/Dialect/CGGI/Transforms/DecomposeOperations.h"

#include <cassert>
#include <cstdint>
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

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DEF_DECOMPOSEOPERATIONS
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

namespace alignment {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "lib/Dialect/CGGI/Transforms/DecomposeOperations.cpp.inc"
}  // namespace alignment

struct ExpandLut2 : public OpRewritePattern<Lut2Op> {
  ExpandLut2(mlir::MLIRContext *context)
      : OpRewritePattern<Lut2Op>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(Lut2Op op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int32_t> coeffs2 = {2, 1};
    auto createLutLinCombOp =
        LutLinCombOp::create(rewriter, op.getLoc(), op.getOutput().getType(),
                             op.getOperands(), coeffs2, op.getLookupTable());
    rewriter.replaceOp(op, createLutLinCombOp);
    return success();
  }
};

struct ExpandLut3 : public OpRewritePattern<Lut3Op> {
  ExpandLut3(mlir::MLIRContext *context)
      : OpRewritePattern<Lut3Op>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(Lut3Op op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int> coeffs3 = {4, 2, 1};
    auto createLutLinCombOp =
        LutLinCombOp::create(rewriter, op.getLoc(), op.getOutput().getType(),
                             op.getOperands(), coeffs3, op.getLookupTable());
    rewriter.replaceOp(op, createLutLinCombOp);
    return success();
  }
};

struct ExpandLutLinComb : public OpRewritePattern<LutLinCombOp> {
  ExpandLutLinComb(mlir::MLIRContext *context)
      : OpRewritePattern<LutLinCombOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(LutLinCombOp op,
                                PatternRewriter &rewriter) const override {
    Type scalarTy = rewriter.getIntegerType(
        cast<lwe::NewLWECiphertextType>(op.getInputs().front().getType())
            .getPlaintextSpace()
            .getRing()
            .getCoefficientType()
            .getIntOrFloatBitWidth());
    // Use LWE operations to create the linear combination of inputs.
    Value result;
    for (auto [input, coeff] :
         llvm::zip(op.getInputs(), op.getCoefficients())) {
      auto scaled = lwe::MulScalarOp::create(
          rewriter, op.getLoc(), input,
          arith::ConstantOp::create(rewriter, op.getLoc(),
                                    rewriter.getIntegerAttr(scalarTy, coeff)));
      result = result
                   ? lwe::AddOp::create(rewriter, op.getLoc(), result, scaled)
                         .getResult()
                   : scaled.getResult();
    }
    rewriter.replaceOpWithNewOp<ProgrammableBootstrapOp>(op, result,
                                                         op.getLookupTable());
    return success();
  }
};

struct DecomposeOperations
    : impl::DecomposeOperationsBase<DecomposeOperations> {
  using DecomposeOperationsBase::DecomposeOperationsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    // Add patterns generated from DRR
    alignment::populateWithGenerated(patterns);

    patterns.add<ExpandLut2, ExpandLut3>(context);
    if (expandLincomb) {
      patterns.add<ExpandLutLinComb>(context);
    }

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
