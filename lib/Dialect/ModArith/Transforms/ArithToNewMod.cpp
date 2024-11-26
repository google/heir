#include "lib/Dialect/ModArith/Transforms/ArithToNewMod.h"

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "arith-new-mod"

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DEF_ARITHTONEWMOD
#include "lib/Dialect/ModArith/Transforms/ArithToNewMod.h.inc"

struct ConvertToAdd : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    LLVM_DEBUG({ llvm::dbgs() << "################### Found one  mult:\n"; });

    // auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto add =
        b.create<mod_arith::AddOp>(op.getLoc(), op.getLhs(), op.getRhs());
    // auto remu = b.create<arith::RemUIOp>(add, cmod);

    rewriter.replaceOp(op, add);
    return success();
  }
};

struct ArithToNewMod : impl::ArithToNewModBase<ArithToNewMod> {
  using ArithToNewModBase::ArithToNewModBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // FIXME: implement pass
    patterns.add<ConvertToAdd>(context);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
