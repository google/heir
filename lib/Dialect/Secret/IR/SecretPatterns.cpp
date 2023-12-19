#include "include/Dialect/Secret/IR/SecretPatterns.h"

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineValueMap.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

namespace {

llvm::SmallVector<Value> buildResolvedIndices(Operation *op,
                                              OperandRange opIndices,
                                              PatternRewriter &rewriter) {
  affine::MemRefAccess access(op);
  affine::AffineValueMap thisMap;
  access.getAccessMap(&thisMap);
  llvm::SmallVector<Value> indices;
  auto indicesIt = opIndices.begin();
  for (unsigned i = 0; i < access.getRank(); ++i) {
    auto affineValue = thisMap.getResult(i);
    if (affineValue.getKind() == AffineExprKind::Constant) {
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(
          op->getLoc(), cast<AffineConstantExpr>(affineValue).getValue()));
    } else {
      assert(affineValue.getKind() == AffineExprKind::DimId &&
             "expected dimensional id or constant for affine operation index");
      indices.push_back(*(indicesIt++));
    }
  }
  return indices;
}

}  // namespace

LogicalResult CollapseSecretlessGeneric::matchAndRewrite(
    GenericOp op, PatternRewriter &rewriter) const {
  for (Type ty : op.getOperandTypes()) {
    if (dyn_cast<SecretType>(ty)) {
      return failure();
    }
  }

  // Allocation ops cannot be collapsed because they may be used in later
  // generics. We should eventually do a proper dataflow analysis to ensure
  // they can be collapsed when no secret data is added to them.
  //
  // There is no good way to identify an allocation op in general. Maybe we can
  // upstream a trait for this?
  for ([[maybe_unused]] const auto op : op.getOps<memref::AllocOp>()) {
    return failure();
  }

  YieldOp yieldOp = op.getYieldOp();
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

LogicalResult RemoveUnusedYieldedValues::matchAndRewrite(
    GenericOp op, PatternRewriter &rewriter) const {
  SmallVector<Value> valuesToRemove;
  for (auto &opOperand : op.getYieldOp()->getOpOperands()) {
    Value result = op.getResults()[opOperand.getOperandNumber()];
    if (result.use_empty()) {
      valuesToRemove.push_back(opOperand.get());
    }
  }

  if (!valuesToRemove.empty()) {
    SmallVector<Value> remainingResults;
    auto modifiedGeneric =
        op.removeYieldedValues(valuesToRemove, rewriter, remainingResults);
    rewriter.replaceAllUsesWith(remainingResults, modifiedGeneric.getResults());
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

LogicalResult RemoveNonSecretGenericArgs::matchAndRewrite(
    GenericOp op, PatternRewriter &rewriter) const {
  bool deletedAny = false;
  for (unsigned i = 0; i < op->getNumOperands(); i++) {
    if (!isa<SecretType>(op->getOperand(i).getType())) {
      deletedAny = true;
      Block *body = op.getBody();
      BlockArgument correspondingArg = body->getArgument(i);

      rewriter.replaceAllUsesWith(correspondingArg, op->getOperand(i));
      rewriter.updateRootInPlace(op, [&]() {
        body->eraseArgument(i);
        op.getOperation()->eraseOperand(i);
      });
      i--;
    }
  }

  return deletedAny ? success() : failure();
}

LogicalResult CaptureAmbientScope::matchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  Value *foundValue = nullptr;
  rewriter.setInsertionPointToStart(genericOp.getBody());
  genericOp.getBody()->walk([&](Operation *op) -> WalkResult {
    for (Value operand : op->getOperands()) {
      Region *operandRegion = operand.getParentRegion();
      if (operandRegion && !genericOp.getRegion().isAncestor(operandRegion)) {
        foundValue = &operand;
        // If this is an index operand of an affine operation, update to memref
        // ops. Affine dimensions must be block arguments for affine.for or
        // affine.parallel.
        if (isa<IndexType>(operand.getType())) {
          llvm::TypeSwitch<Operation *>(op)
              .Case<affine::AffineLoadOp>([&](affine::AffineLoadOp op) {
                rewriter.replaceOp(op, rewriter.create<memref::LoadOp>(
                                           op->getLoc(), op.getMemref(),
                                           buildResolvedIndices(
                                               op, op.getIndices(), rewriter)));
              })
              .Case<affine::AffineStoreOp>([&](affine::AffineStoreOp op) {
                rewriter.replaceOp(
                    op,
                    rewriter.create<memref::StoreOp>(
                        op->getLoc(), op.getValueToStore(), op.getMemref(),
                        buildResolvedIndices(op, op.getIndices(), rewriter)));
              });
        }
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (foundValue == nullptr) {
    return failure();
  }

  Value value = *foundValue;
  rewriter.updateRootInPlace(genericOp, [&]() {
    BlockArgument newArg =
        genericOp.getBody()->addArgument(value.getType(), genericOp.getLoc());
    rewriter.replaceUsesWithIf(value, newArg, [&](mlir::OpOperand &operand) {
      return operand.getOwner()->getParentOp() == genericOp;
    });
    genericOp.getInputsMutable().append(value);
  });

  return success();
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir
