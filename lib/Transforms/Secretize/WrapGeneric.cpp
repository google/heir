#include <iostream>

#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "include/Transforms/Secretize/Passes.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_WRAPGENERIC
#include "include/Transforms/Secretize/Passes.h.inc"

struct WrapWithGeneric : public OpRewritePattern<func::FuncOp> {
  WrapWithGeneric(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<func::FuncOp>(context) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    bool hasSecrets = false;

    SmallVector<Type, 4> newInputs;
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      auto argTy = op.getArgument(i).getType();
      if (op.getArgAttr(i, secret::SecretDialect::kArgSecretAttrName) !=
          nullptr) {
        hasSecrets = true;
        op.removeArgAttr(i, secret::SecretDialect::kArgSecretAttrName);

        auto newTy = secret::SecretType::get(argTy);
        op.getArgument(i).setType(newTy);  // Updates the block argument type.
        newInputs.push_back(newTy);
      } else {
        newInputs.push_back(argTy);
      }
    }

    if (!hasSecrets) {
      // Match failure, no secret inputs.
      return failure();
    }

    auto newOutputs = llvm::to_vector<6>(llvm::map_range(
        op.getResultTypes(),
        [](Type t) -> Type { return secret::SecretType::get(t); }));

    op.setFunctionType(
        FunctionType::get(getContext(), {newInputs}, {newOutputs}));

    // Create a secret.generic op and pull the original function block in.
    Block &opEntryBlock = op.getRegion().front();
    rewriter.setInsertionPointToStart(&opEntryBlock);

    SmallVector<Operation *> opsToErase;
    auto newGeneric = rewriter.create<secret::GenericOp>(
        op.getLoc(), op.getArguments(), newOutputs,
        [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
          //  Map the input values to the block arguments.
          IRMapping mp;
          for (unsigned i = 0; i < blockArguments.size(); ++i) {
            mp.map(opEntryBlock.getArgument(i), blockArguments[i]);
          }

          opEntryBlock.walk([&](Operation *innerOp) {
            if (!isa<func::ReturnOp>(innerOp)) {
              b.clone(*innerOp, mp);
              opsToErase.push_back(innerOp);
            }
          });

          auto *returnOp = opEntryBlock.getTerminator();
          b.create<secret::YieldOp>(
              loc, llvm::to_vector(llvm::map_range(
                       returnOp->getOperands(),
                       [&](Value v) { return mp.lookupOrDefault(v); })));
        });

    rewriter.replaceOp(
        opEntryBlock.getTerminator(),
        rewriter.create<func::ReturnOp>(op.getLoc(), newGeneric.getResults()));

    for (auto *erase : llvm::reverse(opsToErase)) {
      rewriter.eraseOp(erase);
    }

    return success();
  }
};

struct WrapGeneric : impl::WrapGenericBase<WrapGeneric> {
  using WrapGenericBase::WrapGenericBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<WrapWithGeneric>(context);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
