#include "lib/Conversion/BGVToLWE/BGVToLWE.h"

#include <cstddef>
#include <optional>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  //from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOLWE
#include "lib/Conversion/BGVToLWE/BGVToLWE.h.inc"

template <typename BGVOp, typename LWEOp>
struct Convert : public OpRewritePattern<BGVOp> {
  Convert(mlir::MLIRContext *context) : OpRewritePattern<BGVOp>(context) {}

  LogicalResult matchAndRewrite(BGVOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LWEOp>(op, op->getOperands(), op->getAttrs());
    return success();
  }
};

struct BGVToLWE : public impl::BGVToLWEBase<BGVToLWE> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    patterns
        .add<Convert<AddOp, lwe::RAddOp>, Convert<SubOp, lwe::RSubOp>,
             Convert<NegateOp, lwe::RNegateOp>, Convert<MulOp, lwe::RMulOp> >(
            context);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::bgv
