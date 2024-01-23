#include "include/Dialect/Secret/IR/SecretPatterns.h"

#include <cassert>
#include <optional>

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"   // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineValueMap.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

#define DEBUG_TYPE "secret-patterns"

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
      rewriter.modifyOpInPlace(op, [&]() {
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
      rewriter.modifyOpInPlace(op, [&]() {
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
  rewriter.modifyOpInPlace(genericOp, [&]() {
    BlockArgument newArg =
        genericOp.getBody()->addArgument(value.getType(), genericOp.getLoc());
    rewriter.replaceUsesWithIf(value, newArg, [&](mlir::OpOperand &operand) {
      return operand.getOwner()->getParentOp() == genericOp;
    });
    genericOp.getInputsMutable().append(value);
  });

  return success();
}

LogicalResult MergeAdjacentGenerics::matchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  GenericOp nextGenericOp = dyn_cast<GenericOp>(genericOp->getNextNode());
  if (!nextGenericOp) {
    return failure();
  }

  SmallVector<Type> newResultTypes;
  newResultTypes.reserve(genericOp->getNumResults() +
                         nextGenericOp->getNumResults());
  newResultTypes.append(genericOp->getResultTypes().begin(),
                        genericOp->getResultTypes().end());
  newResultTypes.append(nextGenericOp->getResultTypes().begin(),
                        nextGenericOp->getResultTypes().end());

  llvm::SmallVector<Value, 6> newOperands;
  newOperands.append(genericOp->getOperands().begin(),
                     genericOp->getOperands().end());
  // We need to merge the two operand regions and keep track of the mapping
  // from the second generic's operands to the new operands. In the case that
  // the old operand is a result of the first generic, we don't need to include
  // it because the ops that use that operand will be moved inside the first
  // generic.
  DenseMap<Value, int> oldOperandsToNewOperandIndex;
  for (int i = 0; i < nextGenericOp->getNumOperands(); ++i) {
    auto currOperand = nextGenericOp->getOperand(i);
    LLVM_DEBUG(llvm::dbgs() << "Trying to dedupe operand " << i << " in "
                            << *nextGenericOp << "\n");
    std::optional<int> resultIndex = genericOp.findResultIndex(currOperand);
    if (resultIndex.has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "It's result " << resultIndex.value()
                              << " of the previous generic.\n");
      continue;
    }
    bool found = false;
    for (int j = 0; j < newOperands.size(); ++j) {
      if (currOperand == newOperands[j]) {
        LLVM_DEBUG(llvm::dbgs() << "Mapping to operand " << j << "\n");
        oldOperandsToNewOperandIndex[currOperand] = j;
        found = true;
        break;
      }
    }

    if (!found) {
      newOperands.push_back(currOperand);
      oldOperandsToNewOperandIndex[currOperand] = newOperands.size() - 1;
      LLVM_DEBUG(llvm::dbgs() << "Not found, adding to operands at index "
                              << newOperands.size() - 1 << "\n");
    }
  }

  auto newGeneric = rewriter.create<GenericOp>(
      genericOp.getLoc(), newOperands, newResultTypes,
      [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
        IRMapping mp;
        for (BlockArgument blockArg : genericOp.getBody()->getArguments()) {
          mp.map(blockArg, blockArguments[blockArg.getArgNumber()]);
        }

        SmallVector<YieldOp> clonedYieldOps;
        auto &firstOps = genericOp.getBody()->getOperations();
        for (auto &op : firstOps) {
          auto *newOp = b.clone(op, mp);
          if (auto clonedYield = dyn_cast<YieldOp>(newOp)) {
            clonedYieldOps.push_back(clonedYield);
          }
        }

        // For each block argument of the next generic, see if it's a result
        // of the first generic, and if so, map it to the yielded value of
        // the first generic. Otherwise, map it to the corresponding block
        // argument of the new generic.
        for (BlockArgument blockArg : nextGenericOp.getBody()->getArguments()) {
          OpOperand *correspondingOperand =
              nextGenericOp.getOpOperandForBlockArgument(blockArg);
          std::optional<int> resultIndex =
              genericOp.findResultIndex(correspondingOperand->get());
          if (resultIndex.has_value()) {
            mp.map(blockArg,
                   clonedYieldOps[0]->getOperand(resultIndex.value()));
            continue;
          }

          LLVM_DEBUG(
              llvm::dbgs()
              << "Mapping " << blockArg << " in " << *nextGenericOp
              << " to new block arg "
              << oldOperandsToNewOperandIndex[correspondingOperand->get()]
              << "\n");
          mp.map(blockArg, blockArguments[oldOperandsToNewOperandIndex.lookup(
                               correspondingOperand->get())]);
        }

        auto &secondOps = nextGenericOp.getBody()->getOperations();
        for (auto &op : secondOps) {
          auto *newOp = b.clone(op, mp);
          if (auto clonedYield = dyn_cast<YieldOp>(newOp)) {
            clonedYieldOps.push_back(clonedYield);
          }
        }

        // We have cloned two yields, and now we need to merge them.
        assert(clonedYieldOps.size() == 2 &&
               "Expected two yields in cloned generic");
        SmallVector<Value> newYields;
        newYields.reserve(newResultTypes.size());
        for (auto yieldOp : clonedYieldOps) {
          newYields.append(yieldOp->getOperands().begin(),
                           yieldOp->getOperands().end());
          rewriter.eraseOp(yieldOp);
        }
        b.create<YieldOp>(loc, newYields);
      });

  SmallVector<Value> valuesReplacingSecondGeneric;
  valuesReplacingSecondGeneric.reserve(nextGenericOp->getNumResults());
  valuesReplacingSecondGeneric.append(
      newGeneric.getResults().begin() + genericOp->getNumResults(),
      newGeneric.getResults().end());
  rewriter.replaceOp(nextGenericOp, valuesReplacingSecondGeneric);

  SmallVector<Value> valuesReplacingFirstGeneric;
  valuesReplacingFirstGeneric.reserve(genericOp->getNumResults());
  valuesReplacingFirstGeneric.append(
      newGeneric.getResults().begin(),
      newGeneric.getResults().begin() + genericOp->getNumResults());
  rewriter.replaceOp(genericOp, valuesReplacingFirstGeneric);

  return success();
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir
