#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_WRAPGENERIC
#include "lib/Transforms/Secretize/Passes.h.inc"

struct WrapWithGeneric : public OpRewritePattern<func::FuncOp> {
  WrapWithGeneric(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<func::FuncOp>(context) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    bool hasSecrets = false;
    bool secretHasUse = false;

    // check if any of the arguments are secret
    // and if any of the secret arguments are used
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      if (op.getArgAttr(i, secret::SecretDialect::kArgSecretAttrName) !=
          nullptr) {
        hasSecrets = true;

        if (!op.getArgument(i).getUses().empty()) {
          secretHasUse = true;
        }
      }
    }

    if (!hasSecrets) {
      // Match failure, no secret inputs.
      return failure();
    }
    if (!secretHasUse) {
      // No need to wrap the function, no secret inputs are used.
      // remove secret attribute
      for (unsigned i = 0; i < op.getNumArguments(); i++) {
        if (op.getArgAttr(i, secret::SecretDialect::kArgSecretAttrName) !=
            nullptr) {
          op.removeArgAttr(i, secret::SecretDialect::kArgSecretAttrName);
        }
      }
      return failure();
    }

    // begin transform
    SmallVector<Type, 4> newInputs;
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      auto argTy = op.getArgument(i).getType();
      if (op.getArgAttr(i, secret::SecretDialect::kArgSecretAttrName) !=
          nullptr) {
        op.removeArgAttr(i, secret::SecretDialect::kArgSecretAttrName);

        auto newTy = secret::SecretType::get(argTy);
        op.getArgument(i).setType(newTy);  // Updates the block argument type.
        newInputs.push_back(newTy);
      } else {
        newInputs.push_back(argTy);
      }
    }

    auto newOutputs = llvm::to_vector<6>(llvm::map_range(
        op.getResultTypes(),
        [](Type t) -> Type { return secret::SecretType::get(t); }));

    op.setFunctionType(
        FunctionType::get(getContext(), {newInputs}, {newOutputs}));

    // Create a new block where we will insert the new secret.generic and move
    // the function ops into.
    Block &opEntryBlock = op.getRegion().front();
    auto newBlock = rewriter.createBlock(
        &opEntryBlock, opEntryBlock.getArgumentTypes(),
        SmallVector<Location>(opEntryBlock.getNumArguments(), op.getLoc()));

    rewriter.setInsertionPointToStart(newBlock);
    auto newGeneric = rewriter.create<secret::GenericOp>(
        op.getLoc(), op.getArguments(), newOutputs,
        [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
          //  Map the input values to the block arguments.
          IRMapping mp;
          for (unsigned i = 0; i < blockArguments.size(); ++i) {
            mp.map(opEntryBlock.getArgument(i), blockArguments[i]);
          }

          auto *returnOp = opEntryBlock.getTerminator();
          b.create<secret::YieldOp>(
              loc, llvm::to_vector(llvm::map_range(
                       returnOp->getOperands(),
                       [&](Value v) { return mp.lookupOrDefault(v); })));
          returnOp->erase();
        });

    Block &genericBlock = newGeneric.getRegion().front();
    rewriter.inlineBlockBefore(&opEntryBlock,
                               &genericBlock.getOperations().back(),
                               genericBlock.getArguments());
    rewriter.create<func::ReturnOp>(op.getLoc(), newGeneric.getResults());

    return success();
  }
};

struct WrapGeneric : impl::WrapGenericBase<WrapGeneric> {
  using WrapGenericBase::WrapGenericBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // call CSE and Canonicalize here
    // to avoid secretless function body
    OpPassManager eliminateSecretlessBodyPipeline("builtin.module");
    eliminateSecretlessBodyPipeline.addPass(createCSEPass());
    eliminateSecretlessBodyPipeline.addPass(createCanonicalizerPass());
    (void)runPipeline(eliminateSecretlessBodyPipeline, getOperation());

    mlir::RewritePatternSet patterns(context);
    patterns.add<WrapWithGeneric>(context);
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
