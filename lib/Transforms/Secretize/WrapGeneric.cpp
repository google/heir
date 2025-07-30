#include <utility>

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/Secretize/Passes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
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

    SmallVector<Type, 4> newInputs;
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      auto argTy = op.getArgumentTypes()[i];
      if (op.getArgAttr(i, secret::SecretDialect::kArgSecretAttrName) !=
          nullptr) {
        hasSecrets = true;
        op.removeArgAttr(i, secret::SecretDialect::kArgSecretAttrName);

        auto newTy = secret::SecretType::get(argTy);
        if (!op.isDeclaration())
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

    // modification to function type should go through the rewriter
    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionType(
          FunctionType::get(getContext(), {newInputs}, {newOutputs}));
    });

    // Externally defined functions have no body
    if (op.isDeclaration()) {
      return success();
    }
    // Create a new block where we will insert the new secret.generic and move
    // the function ops into.
    Block &opEntryBlock = op.getRegion().front();
    auto *newBlock = rewriter.createBlock(
        &opEntryBlock, opEntryBlock.getArgumentTypes(),
        SmallVector<Location>(opEntryBlock.getNumArguments(), op.getLoc()));

    rewriter.setInsertionPointToStart(newBlock);
    auto newGeneric = secret::GenericOp::create(
        rewriter, op.getLoc(), op.getArguments(), newOutputs,
        [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
          //  Map the input values to the block arguments.
          IRMapping mp;
          for (unsigned i = 0; i < blockArguments.size(); ++i) {
            mp.map(opEntryBlock.getArgument(i), blockArguments[i]);
          }

          auto *returnOp = opEntryBlock.getTerminator();
          secret::YieldOp::create(b, loc,
                                  llvm::to_vector(llvm::map_range(
                                      returnOp->getOperands(), [&](Value v) {
                                        return mp.lookupOrDefault(v);
                                      })));
          returnOp->erase();
        });

    Block &genericBlock = newGeneric.getRegion().front();
    rewriter.inlineBlockBefore(&opEntryBlock,
                               &genericBlock.getOperations().back(),
                               genericBlock.getArguments());
    func::ReturnOp::create(rewriter, op.getLoc(), newGeneric.getResults());

    return success();
  }
};

struct ConvertFuncCall : public OpRewritePattern<func::CallOp> {
  ConvertFuncCall(mlir::MLIRContext *context, Operation *top)
      : mlir::OpRewritePattern<func::CallOp>(context), top(top) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto module = mlir::cast<ModuleOp>(top);
    auto callee = module.lookupSymbol<func::FuncOp>(op.getCallee());
    if (callee.isDeclaration()) {
      return success();
    }

    SmallVector<Value> newOperands;
    auto funcResultTypes = llvm::to_vector(callee.getResultTypes());

    for (auto i = 0; i != op->getNumOperands(); ++i) {
      auto operand = op.getOperand(i);
      auto funcArgType = callee.getArgumentTypes()[i];
      if (mlir::isa<secret::SecretType>(funcArgType)) {
        auto newOperand =
            secret::ConcealOp::create(rewriter, op.getLoc(), operand);
        newOperands.push_back(newOperand.getResult());
      } else {
        newOperands.push_back(operand);
      }
    }

    auto newOp = func::CallOp::create(rewriter, op->getLoc(), op.getCallee(),
                                      funcResultTypes, newOperands);
    newOp->setAttrs(op->getAttrs());

    for (auto i = 0; i != newOp->getNumResults(); ++i) {
      auto result = op.getResult(i);
      auto newResult = newOp.getResult(i);
      if (mlir::isa<secret::SecretType>(newResult.getType())) {
        newResult = secret::RevealOp::create(rewriter, op.getLoc(), newResult);
      }
      rewriter.replaceAllUsesWith(result, newResult);
    }
    rewriter.eraseOp(op);
    return success();
  }

 private:
  Operation *top;
};

struct WrapGeneric : impl::WrapGenericBase<WrapGeneric> {
  using WrapGenericBase::WrapGenericBase;

  void detectSecretGeneric() {
    bool hasSecretGeneric = false;
    getOperation().walk([&](secret::GenericOp op) { hasSecretGeneric = true; });
    if (!hasSecretGeneric) {
      getOperation().emitWarning(
          "No secret found in the module. Did you forget to annotate "
          "{secret.secret} to the function arguments?");
    }
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<WrapWithGeneric>(context);
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));

    // func.call should be converted after callee func type updated
    mlir::RewritePatternSet patterns2(context);
    patterns2.add<ConvertFuncCall>(context, getOperation());
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns2));

    // warn if no secret.generic found
    detectSecretGeneric();
  }
};

}  // namespace heir
}  // namespace mlir
