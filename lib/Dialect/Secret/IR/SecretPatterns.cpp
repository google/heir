#include "lib/Dialect/Secret/IR/SecretPatterns.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineValueMap.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#define DEBUG_TYPE "secret-patterns"

namespace mlir {
namespace heir {
namespace secret {

namespace {

// Returns whether an operation retrieves constant global data.
bool isConstantGlobal(Operation *op) {
  auto getGlobal = dyn_cast<memref::GetGlobalOp>(op);
  if (!getGlobal) return false;

  auto *symbolTableOp = getGlobal->getParentWithTrait<OpTrait::SymbolTable>();
  if (!symbolTableOp) return false;
  auto global = dyn_cast_or_null<memref::GlobalOp>(
      SymbolTable::lookupSymbolIn(symbolTableOp, getGlobal.getNameAttr()));
  if (!global) return false;

  // Check if the global memref is a constant.
  return isa<DenseElementsAttr>(global.getConstantInitValue());
}

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
      LLVM_DEBUG(llvm::dbgs() << arg << " has no uses; removing\n");
      hasUnusedOps = true;
      rewriter.modifyOpInPlace(op, [&]() {
        body->eraseArgument(i);
        op.getOperation()->eraseOperand(i);
      });
      // Ensure the next iteration uses the right arg number
      --i;
    } else if (llvm::any_of(arg.getUsers(), [&](Operation *user) {
                 return llvm::isa<YieldOp>(user);
               })) {
      LLVM_DEBUG(llvm::dbgs() << arg << " is passed through to yield\n");
      // In this case, the arg is passed through to the yield, and the yield
      // can be removed and replaced with the operand. Note we don't need to
      // remove the block argument itself since a subsequent iteration of this
      // pattern will detect if that is possible (if it has no other uses).
      Value replacementValue = op.getOperand(arg.getArgNumber());
      SmallVector<Value> yieldedValuesToRemove;
      SmallVector<Value> resultsToReplace;
      SmallVector<Value> replacementValues;

      for (auto &opOperand : op.getYieldOp()->getOpOperands()) {
        if (opOperand.get() == arg) {
          yieldedValuesToRemove.push_back(opOperand.get());
          resultsToReplace.push_back(
              op.getResult(opOperand.getOperandNumber()));
          replacementValues.push_back(replacementValue);
        }
      }
      rewriter.replaceAllUsesWith(resultsToReplace, replacementValues);

      SmallVector<Value> remainingResults;
      auto modifiedGeneric = op.removeYieldedValues(yieldedValuesToRemove,
                                                    rewriter, remainingResults);
      rewriter.replaceAllUsesWith(remainingResults,
                                  modifiedGeneric.getResults());
      rewriter.eraseOp(op);
      return success();
    }
  }

  return hasUnusedOps ? success() : failure();
}

LogicalResult RemoveUnusedYieldedValues::matchAndRewrite(
    GenericOp op, PatternRewriter &rewriter) const {
  SmallVector<int> resultIndicesToRemove;
  for (auto &opOperand : op.getYieldOp()->getOpOperands()) {
    Value result = op.getResults()[opOperand.getOperandNumber()];
    if (result.use_empty()) {
      LLVM_DEBUG(op.emitRemark() << " result of generic at index "
                                 << opOperand.getOperandNumber()
                                 << " is unused, removing from yield\n");
      resultIndicesToRemove.push_back(opOperand.getOperandNumber());
    }
  }

  if (!resultIndicesToRemove.empty()) {
    SmallVector<Value> remainingResults;
    auto modifiedGeneric = op.removeYieldedValues(resultIndicesToRemove,
                                                  rewriter, remainingResults);
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
          auto point = rewriter.saveInsertionPoint();
          rewriter.setInsertionPointAfter(op);
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
          rewriter.restoreInsertionPoint(point);
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
      return operand.getOwner()->getParentOfType<secret::GenericOp>() ==
             genericOp;
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

LogicalResult YieldStoredMemrefs::matchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  Value memref;
  auto walkResult = genericOp.getBody()->walk([&](Operation *op) -> WalkResult {
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      memref = storeOp.getMemRef();
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      memref = storeOp.getMemRef();
    }

    if (memref) {
      ValueRange yieldOperands = genericOp.getYieldOp().getOperands();
      int index =
          std::find(yieldOperands.begin(), yieldOperands.end(), memref) -
          yieldOperands.begin();
      if (index < genericOp.getYieldOp().getNumOperands()) {
        // The memref is already yielded
        return WalkResult::advance();
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // interrupt means we found a memref to add to the yield
  if (!walkResult.wasInterrupted()) {
    return failure();
  }

  auto [modifiedGeneric, newResults] =
      genericOp.addNewYieldedValues(ValueRange{memref}, rewriter);
  rewriter.replaceOp(genericOp,
                     ValueRange(modifiedGeneric.getResults().drop_back(1)));
  return success();
}

LogicalResult DedupeYieldedValues::matchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, int, 4> yieldedValueToIndex;
  int indexToRemove = -1;
  int replacementIndex = -1;
  for (auto &opOperand : genericOp.getYieldOp()->getOpOperands()) {
    if (yieldedValueToIndex.contains(opOperand.get())) {
      indexToRemove = opOperand.getOperandNumber();
      replacementIndex = yieldedValueToIndex[opOperand.get()];
      break;
    }
    yieldedValueToIndex[opOperand.get()] = opOperand.getOperandNumber();
  }

  if (indexToRemove < 0) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Found value to dedupe at yield index " << indexToRemove
                 << "\n";
    genericOp.dump();
  });

  Value resultToRemove = genericOp.getResults()[indexToRemove];
  Value replacementResult = genericOp.getResults()[replacementIndex];
  rewriter.replaceAllUsesWith(resultToRemove, replacementResult);

  SmallVector<Value> remainingResults;
  auto modifiedGeneric = genericOp.removeYieldedValues(
      {indexToRemove}, rewriter, remainingResults);
  rewriter.replaceAllUsesWith(remainingResults, modifiedGeneric.getResults());
  rewriter.eraseOp(genericOp);

  LLVM_DEBUG({
    llvm::dbgs() << "After replacing generic\n";
    modifiedGeneric.dump();
  });
  return success();
}

bool HoistOpBeforeGeneric::canHoist(Operation &op, GenericOp genericOp) const {
  bool inConfiguredList =
      std::find(opTypes.begin(), opTypes.end(), op.getName().getStringRef()) !=
      opTypes.end();
  bool allOperandsAreBlockArgsOrAmbient =
      llvm::all_of(op.getOperands(), [&](Value operand) {
        if (isa<BlockArgument>(operand)) {
          // This is a block argument defined in a block that contains the
          // generic.
          return cast<BlockArgument>(operand).getParentRegion()->isAncestor(
              &genericOp.getRegion());
        }
        return operand.getDefiningOp()->getBlock() != op.getBlock();
      });
  return inConfiguredList && allOperandsAreBlockArgsOrAmbient;
}

LogicalResult HoistOpBeforeGeneric::matchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  auto &opRange = genericOp.getBody()->getOperations();
  if (opRange.size() <= 2) {
    // This corresponds to a fixed point of the pattern: if an op is hoisted,
    // it will be in a single-op generic, (yield is the second op), and if
    // that triggers the pattern, it will be an infinite loop.
    return failure();
  }

  auto it = std::find_if(opRange.begin(), opRange.end(), [&](Operation &op) {
    return canHoist(op, genericOp);
  });
  if (it == opRange.end()) {
    return failure();
  }

  Operation *opToHoist = &*it;
  LLVM_DEBUG(llvm::dbgs() << "Hoisting " << *opToHoist << "\n");
  genericOp.extractOpBeforeGeneric(opToHoist, rewriter);
  return success();
}

bool HoistOpAfterGeneric::canHoist(Operation &op) const {
  bool inConfiguredList =
      std::find(opTypes.begin(), opTypes.end(), op.getName().getStringRef()) !=
      opTypes.end();
  bool allUsesAreYields = llvm::all_of(
      op.getUsers(), [&](Operation *user) { return isa<YieldOp>(user); });
  return inConfiguredList && allUsesAreYields;
}

LogicalResult HoistOpAfterGeneric::matchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  auto &opRange = genericOp.getBody()->getOperations();
  if (opRange.size() <= 2) {
    // This corresponds to a fixed point of the pattern: if an op is hoisted,
    // it will be in a single-op generic, (yield is the second op), and if
    // that triggers the pattern, it will be an infinite loop.
    return failure();
  }

  auto it = std::find_if(opRange.begin(), opRange.end(),
                         [&](Operation &op) { return canHoist(op); });
  if (it == opRange.end()) {
    return failure();
  }

  Operation *opToHoist = &*it;
  LLVM_DEBUG(llvm::dbgs() << "Hoisting " << *opToHoist << "\n");

  extractOpAfterGeneric(genericOp, opToHoist, rewriter);
  return success();
}

LogicalResult HoistPlaintextOps::matchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  auto &opRange = genericOp.getBody()->getOperations();
  if (opRange.size() <= 2) {
    // This corresponds to a fixed point of the pattern: if an op is hoisted,
    // it will be in a single-op generic, (yield is the second op), and if
    // that triggers the pattern, it will be an infinite loop.
    //
    // If you encounter this and the op is actually plaintext, you can instead
    // use CollapseSecretlessGeneric and eliminate the generic op entirely.
    return failure();
  }

  auto canHoist = [&](Operation &op) {
    if (isa<YieldOp>(op)) {
      return false;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Considering whether " << op << " can be hoisted\n");
    if (!isSpeculatable(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "Op is not speculatable\n");
      return false;
    }
    for (Value operand : op.getOperands()) {
      auto blockArg = dyn_cast<BlockArgument>(operand);

      if (blockArg) {
        auto owningGeneric =
            dyn_cast<GenericOp>(blockArg.getOwner()->getParentOp());
        bool isEncryptedBlockArg =
            owningGeneric &&
            isa<SecretType>(
                owningGeneric.getOperand(blockArg.getArgNumber()).getType());
        LLVM_DEBUG(llvm::dbgs()
                   << "operand " << operand << " is a "
                   << (isEncryptedBlockArg ? "encrypted" : "plaintext")
                   << " block arg\n");
        if (isEncryptedBlockArg) {
          return false;
        }
      } else {
        bool isPlaintextAmbient =
            operand.getDefiningOp()->getBlock() != op.getBlock() &&
            !mlir::isa<SecretType>(operand.getType());

        LLVM_DEBUG(llvm::dbgs()
                   << "operand " << operand << " is a "
                   << (isPlaintextAmbient ? "plaintext" : "encrypted")
                   << " ambient SSA value\n");
        if (!isPlaintextAmbient) {
          return false;
        }
      }
    }

    return true;
  };

  LLVM_DEBUG(
      llvm::dbgs() << "Scanning generic body looking for ops to hoist...\n");

  // We can't hoist them as they are detected because the process of hoisting
  // alters the context generic op.
  llvm::SmallVector<Operation *> opsToHoist;
  bool hoistedAny = false;
  for (Operation &op : opRange) {
    if (canHoist(op)) {
      opsToHoist.push_back(&op);
      hoistedAny = true;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << opsToHoist.size()
                          << " ops to hoist\n");

  for (Operation *op : opsToHoist) {
    genericOp.extractOpBeforeGeneric(op, rewriter);
  }

  LLVM_DEBUG(llvm::dbgs() << "Done hoisting\n");

  return hoistedAny ? success() : failure();
}

void genericAbsorbConstants(secret::GenericOp genericOp,
                            mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPointToStart(genericOp.getBody());
  genericOp.getBody()->walk([&](Operation *op) -> WalkResult {
    for (Value operand : op->getOperands()) {
      // If this is a block argument, get the generic's corresponding operand.
      Value opOperand = operand;
      auto blockArg = dyn_cast<BlockArgument>(operand);
      bool isGenericBlockArg =
          blockArg && blockArg.getOwner()->getParentOp() == genericOp;
      // Only get the generic's op operand when the block argument belongs to
      // the generic. (Otherwise it could be a contained operations'
      // block argument).
      if (isGenericBlockArg) {
        opOperand = genericOp.getOpOperandForBlockArgument(blockArg)->get();
      }
      auto *definingOp = opOperand.getDefiningOp();
      bool isConstantLike =
          definingOp && (definingOp->hasTrait<OpTrait::ConstantLike>() ||
                         isConstantGlobal(definingOp));
      if (isConstantLike) {
        // If the definingOp is outside of the generic region, then copy it
        // inside the region.
        Region *operandRegion = definingOp->getParentRegion();
        if (operandRegion && !genericOp.getRegion().isAncestor(operandRegion)) {
          auto *copiedOp = rewriter.clone(*definingOp);
          rewriter.replaceAllUsesWith(operand, copiedOp->getResults());
          // If this was a block argument, additionally remove the block
          // argument.
          if (isGenericBlockArg) {
            int index = blockArg.getArgNumber();
            rewriter.modifyOpInPlace(op, [&]() {
              genericOp.getBody()->eraseArgument(index);
              genericOp.getOperation()->eraseOperand(index);
            });
          }
        }
      }
    }
    return WalkResult::advance();
  });
}

LogicalResult extractGenericBody(secret::GenericOp genericOp,
                                 mlir::IRRewriter &rewriter) {
  auto module = genericOp->getParentOfType<ModuleOp>();
  if (!module) {
    return failure();
  }

  SmallVector<Operation *> opsToCopy;
  for (auto &op : genericOp.getBody()->getOperations()) {
    if (isa<secret::YieldOp>(op)) {
      continue;
    }
    opsToCopy.push_back(&op);
  }

  auto yieldOp = genericOp.getYieldOp();
  auto inputTypes = genericOp.getBody()->getArgumentTypes();
  auto inputs = genericOp.getBody()->getArguments();
  auto resultTypes = genericOp.getBody()->getTerminator()->getOperandTypes();

  OpBuilder builder(module);
  builder.setInsertionPoint(genericOp->getParentOfType<func::FuncOp>());
  auto type = builder.getFunctionType(inputTypes, resultTypes);
  std::string funcName = llvm::formatv(
      "internal_generic_{0}", mlir::hash_value(yieldOp.getValues()[0]));
  auto func = builder.create<func::FuncOp>(module.getLoc(), funcName, type);

  // Populate function body by cloning the ops in the inner body and mapping
  // the func args and func outputs.
  Block *block = func.addEntryBlock();
  builder.setInsertionPointToEnd(block);

  // Map the input values to the block arguments.
  IRMapping mp;
  for (int index = 0; index < inputs.size(); ++index) {
    mp.map(inputs[index], block->getArgument(index));
  }

  for (auto &op : opsToCopy) {
    builder.clone(*op, mp);
  }

  auto returnOperands = llvm::to_vector(
      llvm::map_range(yieldOp.getOperands(),
                      [&](Value operand) { return mp.lookup(operand); }));
  builder.create<func::ReturnOp>(func.getLoc(), returnOperands);

  // Call the function.
  builder.setInsertionPointToStart(genericOp.getBody());
  auto callOp = builder.create<func::CallOp>(genericOp.getLoc(), func, inputs);
  rewriter.modifyOpInPlace(
      yieldOp, [&]() { yieldOp->setOperands(callOp.getResults()); });

  for (auto &op : llvm::reverse(opsToCopy)) {
    rewriter.eraseOp(op);
  }

  return success();
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir
