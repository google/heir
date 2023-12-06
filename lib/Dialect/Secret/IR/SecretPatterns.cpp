#include "include/Dialect/Secret/IR/SecretPatterns.h"

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

LogicalResult CollapseSecretlessGeneric::matchAndRewrite(
    GenericOp op, PatternRewriter &rewriter) const {
  for (Type ty : op.getOperandTypes()) {
    if (dyn_cast<SecretType>(ty)) {
      return failure();
    }
  }

  YieldOp yieldOp = dyn_cast<YieldOp>(op.getBody()->getOperations().back());
  rewriter.inlineBlockBefore(op.getBody(), op.getOperation(), op.getInputs());
  rewriter.replaceOp(op, yieldOp.getValues());
  rewriter.eraseOp(yieldOp);
  return success();
};

LogicalResult RemoveUnusedGenericArgs::matchAndRewrite(
    GenericOp op, PatternRewriter &rewriter) const {
  bool hasUnusedOps = false;
  Block *body = op.getBody();
  for (int i = 0; i < body->getArguments().size(); ++i) {
    BlockArgument arg = body->getArguments()[i];
    if (arg.use_empty()) {
      hasUnusedOps = true;
      rewriter.updateRootInPlace(op, [&]() {
        body->eraseArgument(i);
        op.getOperation()->eraseOperand(i);
      });
      // Ensure the next iteration uses the right arg number
      --i;
    }
  }

  return hasUnusedOps ? success() : failure();
}

LogicalResult RemoveNonSecretGenericArgs::matchAndRewrite(
    GenericOp op, PatternRewriter &rewriter) const {
  bool deletedAny = false;
  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<SecretType>(operand.get().getType())) {
      deletedAny = true;
      Block *body = op.getBody();
      BlockArgument correspondingArg =
          body->getArgument(operand.getOperandNumber());

      rewriter.replaceAllUsesWith(correspondingArg, operand.get());
      rewriter.updateRootInPlace(op, [&]() {
        body->eraseArgument(operand.getOperandNumber());
        op.getOperation()->eraseOperand(operand.getOperandNumber());
      });
    }
  }

  return deletedAny ? success() : failure();
}
}  // namespace secret
}  // namespace heir
}  // namespace mlir
