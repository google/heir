#include "include/Dialect/Secret/Transforms/DistributeGeneric.h"

#include <algorithm>
#include <utility>

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretPatterns.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "distribute-generic"

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETDISTRIBUTEGENERIC
#include "include/Dialect/Secret/Transforms/Passes.h.inc"

std::optional<Value> ofrToValue(std::optional<OpFoldResult> ofr) {
  if (ofr.has_value()) {
    if (auto value = llvm::dyn_cast_if_present<Value>(*ofr)) {
      return value;
    }
  }
  return std::nullopt;
}

// Split a secret.generic containing multiple ops into multiple secret.generics.
//
// E.g.,
//
//    %res = secret.generic ins(%value : !secret.secret<i32>) {
//    ^bb0(%clear_value: i32):
//      %c7 = arith.constant 7 : i32
//      %0 = arith.muli %clear_value, %c7 : i32
//      secret.yield %0 : i32
//    } -> (!secret.secret<i32>)
//
// is transformed to
//
//    %secret_7 = secret.generic {
//      %c7 = arith.constant 7 : i32
//      secret.yield %c7 : i32
//    } -> !secret.secret<i32>
//    %1 = secret.generic ins(
//       %arg0, %secret_7 : !secret.secret<i32>, !secret.secret<i32>) {
//    ^bb0(%clear_arg0: i32, %clear_7: i32):
//      %7 = arith.muli %clear_arg0, %clear_7 : i32
//      secret.yield %7 : i32
//    } -> !secret.secret<i32>
//
// When options are provided specifying which ops to distribute, the pattern
// will split at the first detected specified op, possibly creating three new
// secret.generics, and otherwise will split it at the first op from the entry
// block, and will always create two secret.generics.
struct SplitGeneric : public OpRewritePattern<GenericOp> {
  SplitGeneric(mlir::MLIRContext *context,
               llvm::ArrayRef<std::string> opsToDistribute)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1),
        opsToDistribute(opsToDistribute) {}

  void distributeThroughRegionHoldingOp(GenericOp genericOp,
                                        Operation &opToDistribute,
                                        PatternRewriter &rewriter) const {
    assert(opToDistribute.getNumRegions() > 0 &&
           "opToDistribute must have at least one region");
    assert(genericOp.getBody()->getOperations().size() == 2 &&
           "opToDistribute must have one non-yield op");

    // Supports ops with countable loop iterations, like affine.for and scf.for,
    // but not scf.while which has multiple associated regions.
    if (auto loop = dyn_cast<LoopLikeOpInterface>(opToDistribute)) {
      // Example:
      //
      //   secret.generic ins(%value : !secret.secret<...>) {
      //   ^bb0(%clear_value: ...):
      //     %1 = scf.for ... iter_args(%iter_arg = %clear_value) -> ... {
      //       scf.yield ...
      //     }
      //     secret.yield %1 : ...
      //   } -> (!secret.secret<...>)
      //
      // This needs to be converted to:
      //
      //   %1 = scf.for ... iter_args(%iter_arg = %value) -> ... {
      //     %2 = secret.generic ins(%iter_arg : !secret.secret<...>) {
      //     ^bb0(%clear_iter_arg: ...):
      //       ...
      //       secret.yield %1 : ...
      //     }
      //     scf.yield %2 : ...
      //   }
      //
      // Terminators of the region are not part of the secret, since they just
      // handle control flow.

      // Before moving the loop out of the generic, connect the loop's operands
      // to the corresponding secret operands (via the block argument number).
      rewriter.startRootUpdate(genericOp);

      // Set the loop op's operands that came from the secret generic block
      // to be the the corresponding operand of the generic op.
      for (OpOperand &operand : opToDistribute.getOpOperands()) {
        OpOperand *corrGenericOperand =
            genericOp.getOpOperandForBlockArgument(operand.get());
        if (corrGenericOperand != nullptr) {
          operand.set(corrGenericOperand->get());
        }
      }

      // Set the op's region iter arg types, which need to match the possibly
      // new type of the operands modified above
      for (auto [arg, operand] :
           llvm::zip(loop.getRegionIterArgs(), loop.getInits())) {
        arg.setType(operand.getType());
      }

      opToDistribute.moveBefore(genericOp);
      // Now the loop is before the secret generic, but the generic still
      // yields the loop's result (the loop should yield the generic's result)
      // and the generic's body still needs to be moved inside the loop.

      // Before touching the loop body, make a list of all its non-terminator
      // ops for later moving.
      auto &loopBodyBlocks = loop.getLoopRegions().front()->getBlocks();
      SmallVector<Operation *> loopBodyOps;
      for (Operation &op : loopBodyBlocks.begin()->without_terminator()) {
        loopBodyOps.push_back(&op);
      }

      // Move the generic op to be the first op of the loop body.
      genericOp->moveBefore(&loopBodyBlocks.front().getOperations().front());

      // Update the yielded values by the terminators of the two ops' blocks.
      ValueRange yieldedValues = loop.getYieldedValues();
      genericOp.getYieldOp()->setOperands(yieldedValues);
      // An affine.for op might not have a yielded value, and only manipulate
      // memrefs in its body. In this case, the secret.generic may still yield
      // memrefs, but the affine.for will yield nothing.
      if (!yieldedValues.empty()) {
        auto *terminator = opToDistribute.getRegion(0).front().getTerminator();
        terminator->setOperands(genericOp.getResults());
      }

      // Update the return type of the loop op to match its terminator.
      auto resultRange = loop.getLoopResults();
      if (resultRange.has_value()) {
        for (auto [result, yielded] :
             llvm::zip(resultRange.value(), yieldedValues)) {
          result.setType(yielded.getType());
        }
      }

      // Move the old loop body ops into the secret.generic
      for (auto *op : loopBodyOps) {
        op->moveBefore(genericOp.getYieldOp());
      }

      // One of the secret.generic's inputs may still refer to the loop's
      // iter_args initializer, when now it should refer to the iter_arg itself.
      for (OpOperand &operand : genericOp->getOpOperands()) {
        for (auto [iterArg, iterArgInit] :
             llvm::zip(loop.getRegionIterArgs(), loop.getInits())) {
          if (operand.get() == iterArgInit) operand.set(iterArg);
        }
      }

      // The ops within the secret generic may still refer to the loop
      // iter_args, which are not part of of the secret.generic's block. To be
      // a bit more general, walk the entire generic body, and for any operand
      // not in the block, add it as an operand to the secret.generic.
      Block *genericBlock = genericOp.getBody();
      genericBlock->walk([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (operand.getParentBlock() != genericBlock) {
            if (isa<SecretType>(operand.getType())) {
              LLVM_DEBUG({
                llvm::dbgs() << "Found an operand that is secret, but not part "
                                "of the generic: "
                             << operand << "\n";
              });
              // Find the secret.generic operand that corresponds to this
              // operand
              int operandNumber =
                  std::find(genericOp.getOperands().begin(),
                            genericOp.getOperands().end(), operand) -
                  genericOp.getOperands().begin();
              assert(operandNumber < genericOp.getNumOperands() &&
                     "operand not found in secret.generic");
              BlockArgument blockArg = genericBlock->getArgument(operandNumber);
              operand.replaceUsesWithIf(blockArg, [&](OpOperand &use) {
                return use.getOwner()->getParentOp() == genericOp;
              });
            }
          }
        }
      });

      // Finally, ops that came after the original secret.generic may still
      // refer to a secret.generic result, when now they should refer to the
      // corresponding result of the loop, if the loop has results.
      for (OpResult genericResult : genericOp.getResults()) {
        if (loop.getLoopResults().has_value()) {
          auto correspondingLoopResult =
              loop.getLoopResults().value()[genericResult.getResultNumber()];
          genericResult.replaceUsesWithIf(
              correspondingLoopResult, [&](OpOperand &use) {
                return use.getOwner()->getParentOp() != loop.getOperation();
              });
        }
      }

      rewriter.finalizeRootUpdate(genericOp);
      return;
    }

    // TODO(https://github.com/google/heir/issues/307): handle
    // RegionBranchOpInterface (scf.while, scf.if).
  }

  /// Move an op from the body of one secret.generic to an earlier
  /// secret.generic in the same block. Updates the yielded values and operands
  /// of the secret.generics appropriately.
  ///
  /// `opToMove` must not depend on the results of any other ops in the source
  /// generic, only its block arguments.
  ///
  /// Returns a new version of the targetGeneric, which replaces the input
  /// `targetGeneric`.
  GenericOp moveOpToEarlierGeneric(Operation &opToMove, GenericOp sourceGeneric,
                                   GenericOp targetGeneric,
                                   PatternRewriter &rewriter) const {
    LLVM_DEBUG(opToMove.emitRemark() << "Moving op to earlier generic\n");

    assert(opToMove.getParentOp() == sourceGeneric &&
           "opToMove must be in sourceGeneric");
    assert(sourceGeneric->getBlock() == targetGeneric->getBlock() &&
           "source and target generics must be in the same block");

    IRMapping cloningMp;
    for (OpOperand &operand : opToMove.getOpOperands()) {
      if (operand.get().getParentBlock() == sourceGeneric.getBody()) {
        LLVM_DEBUG(llvm::dbgs() << "opToMove depends on block argument "
                                << operand.get() << " of source generic\n");
        assert(operand.get().isa<BlockArgument>() &&
               "opToMove has a non-block-argument operand defined in the "
               "source generic");

        BlockArgument blockArg = cast<BlockArgument>(operand.get());
        Value sourceGenericArg =
            sourceGeneric.getOperand(blockArg.getArgNumber());
        LLVM_DEBUG(opToMove.emitRemark()
                   << "Moving op requires adding " << sourceGenericArg
                   << " to target generic\n");

        // The operand may be the result of the generic we'd like to move it to,
        // in which case the targetGeneric's result corresponds to a yielded
        // value in targetGeneric, and we can map the op's operand to that
        // yielded value.
        auto *definingOp = sourceGenericArg.getDefiningOp();
        if (definingOp == targetGeneric.getOperation()) {
          int resultIndex =
              std::find(targetGeneric.getResults().begin(),
                        targetGeneric.getResults().end(), sourceGenericArg) -
              targetGeneric.getResults().begin();

          assert(resultIndex >= 0 && "unable to find result in yield");
          Value yieldedValue =
              targetGeneric.getYieldOp()->getOperand(resultIndex);
          LLVM_DEBUG(llvm::dbgs()
                     << "Mapping " << operand.get()
                     << " to existing yielded value " << yieldedValue << "\n");
          cloningMp.map(operand.get(), yieldedValue);
          continue;
        }

        // The operand may correspond to an existing input to the secret
        // generic, in which case we don't need to add a new value.
        int foundArgIndex =
            std::find(targetGeneric.getOperands().begin(),
                      targetGeneric.getOperands().end(), sourceGenericArg) -
            targetGeneric.getOperands().begin();
        if (foundArgIndex < targetGeneric.getOperands().size()) {
          BlockArgument existingArg =
              targetGeneric.getBody()->getArgument(foundArgIndex);
          LLVM_DEBUG(llvm::dbgs() << "Mapping " << operand.get()
                                  << " to existing targetGeneric block arg: "
                                  << existingArg << "\n");
          cloningMp.map(operand.get(), existingArg);
          continue;
        }

        // Otherwise, the operand must be an input to the sourceGeneric, which
        // must be added to the targetGeneric.
        targetGeneric.getInputsMutable().append(sourceGenericArg);
        BlockArgument newBlockArg = targetGeneric.getBody()->addArgument(
            operand.get().getType(), targetGeneric.getLoc());
        LLVM_DEBUG(llvm::dbgs() << "Mapping " << operand.get()
                                << " to new block arg added to targetGeneric "
                                << newBlockArg << "\n");
        cloningMp.map(operand.get(), newBlockArg);
      }

      // If the operand is not a sourceGeneric block argument, then it must be
      // a value defined in the enclosing scope. It cannot be a value defined
      // between the two secret.generics, because in this pattern we only
      // invoke this just after splitting a generic into two adjacent ops.
      assert(isa<BlockArgument>(operand.get()) ||
             operand.get().getDefiningOp()->isBeforeInBlock(targetGeneric) &&
                 "Invalid use of moveOpToEarlierGeneric");
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Cloning " << opToMove << " to target generic\n");
    Operation *clonedOp = rewriter.clone(opToMove, cloningMp);
    clonedOp->moveBefore(targetGeneric.getYieldOp());
    auto [modifiedGeneric, newResults] =
        targetGeneric.addNewYieldedValues(clonedOp->getResults(), rewriter);
    rewriter.replaceOp(
        targetGeneric,
        ValueRange(modifiedGeneric.getResults().drop_back(newResults.size())));
    LLVM_DEBUG(modifiedGeneric.emitRemark()
               << "Added new yielded values to target generic\n");

    // Finally, add the new targetGeneric results to the sourceGeneric, and
    // replace the opToMove with the new block arguments.
    sourceGeneric.getInputsMutable().append(newResults);
    SmallVector<Location, 1> newLocs(newResults.size(), sourceGeneric.getLoc());
    auto clearTypes = llvm::to_vector<6>(llvm::map_range(
        newResults.getTypes(),
        [](Type t) -> Type { return cast<SecretType>(t).getValueType(); }));
    SmallVector<BlockArgument> newBlockArgs = llvm::to_vector<6>(
        sourceGeneric.getBody()->addArguments(clearTypes, newLocs));

    rewriter.replaceOp(&opToMove, ValueRange{newBlockArgs});
    LLVM_DEBUG(sourceGeneric.emitRemark()
               << "Added new operands from targetGeneric\n");

    return modifiedGeneric;
  }

  /// Split a secret.generic at a given op, creating two secret.generics where
  /// the first contains the ops preceding `opToDistribute`, and the second
  /// starts with `opToDistribute`, which can then be given as input to
  /// `splitGenericAfterFirstOp`.
  void splitGenericBeforeOp(GenericOp genericOp, Operation &stopBefore,
                            PatternRewriter &rewriter) const {
    LLVM_DEBUG(genericOp.emitRemark()
               << " splitting generic before op " << stopBefore << "\n");

    auto newGeneric = splitGenericAfterFirstOp(genericOp, rewriter);
    while (&genericOp.getBody()->getOperations().front() != &stopBefore) {
      auto &op = genericOp.getBody()->getOperations().front();
      newGeneric = moveOpToEarlierGeneric(op, genericOp, newGeneric, rewriter);
    }
  }

  // Splits a generic op after a given opToDistribute. A newly created GenericOp
  // contains the opToDistribute
  GenericOp splitGenericAfterFirstOp(GenericOp genericOp,
                                     PatternRewriter &rewriter) const {
    Operation &firstOp = genericOp.getBody()->front();
    LLVM_DEBUG(firstOp.emitRemark() << " splitting generic after this op\n");

    // Result types are secret versions of the results of the op, since the
    // secret will yield all of this op's results immediately.
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(firstOp.getNumResults());
    for (Type ty : firstOp.getResultTypes()) {
      newResultTypes.push_back(SecretType::get(ty));
    }

    auto newGeneric = rewriter.create<GenericOp>(
        genericOp.getLoc(), genericOp.getInputs(), newResultTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
          IRMapping mp;
          for (BlockArgument blockArg : genericOp.getBody()->getArguments()) {
            mp.map(blockArg, blockArguments[blockArg.getArgNumber()]);
          }
          auto *newOp = b.clone(firstOp, mp);
          b.create<YieldOp>(loc, newOp->getResults());
        });

    LLVM_DEBUG(newGeneric.emitRemark() << " created new generic op\n");

    // Once the op is split off into a new generic op, we need to add new
    // operands to the old generic op, add new corresponding block arguments,
    // and replace all uses of the opToDistribute's results with the created
    // block arguments.
    SmallVector<Value> oldGenericNewBlockArgs;
    rewriter.updateRootInPlace(genericOp, [&]() {
      genericOp.getInputsMutable().append(newGeneric.getResults());
      for (auto ty : firstOp.getResultTypes()) {
        BlockArgument arg =
            genericOp.getBody()->addArgument(ty, firstOp.getLoc());
        oldGenericNewBlockArgs.push_back(arg);
      }
    });
    rewriter.replaceOp(&firstOp, oldGenericNewBlockArgs);

    return newGeneric;
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    Block *body = op.getBody();
    unsigned numOps = body->getOperations().size();
    assert(numOps > 0 &&
           "secret.generic must have nonempty body (the verifier should "
           "enforce this)");

    // Recursive base case: stop if there's only one op left, and it has no
    // regions, noting that we check for 2 ops because the last op is enforced
    // to be a yield op by the verifier.
    if (numOps == 2 && body->front().getRegions().empty()) {
      return failure();
    }

    Operation *opToDistribute = nullptr;
    bool first = true;
    if (opsToDistribute.empty()) {
      opToDistribute = &body->front();
    } else {
      for (Operation &op : body->getOperations()) {
        // op.getName().getStringRef() is the qualified op name (e.g.,
        // affine.for)
        if (std::find(opsToDistribute.begin(), opsToDistribute.end(),
                      op.getName().getStringRef()) != opsToDistribute.end()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Found op to distribute: " << op.getName() << "\n");
          opToDistribute = &op;
          break;
        }
        first = false;
      }
    }

    // Base case: if none of a generic op's member ops are in the list of ops
    // to process, stop.
    if (opToDistribute == nullptr) return failure();

    if (numOps == 2 && !opToDistribute->getRegions().empty()) {
      LLVM_DEBUG(opToDistribute->emitRemark()
                 << "Distributing through region holding op isolated in its "
                    "own generic\n");
      distributeThroughRegionHoldingOp(op, *opToDistribute, rewriter);
      return success();
    }

    if (first) {
      splitGenericAfterFirstOp(op, rewriter);
    } else {
      splitGenericBeforeOp(op, *opToDistribute, rewriter);
    }

    return success();
  }

 private:
  llvm::ArrayRef<std::string> opsToDistribute;
};

struct DistributeGeneric
    : impl::SecretDistributeGenericBase<DistributeGeneric> {
  using SecretDistributeGenericBase::SecretDistributeGenericBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    LLVM_DEBUG({
      llvm::dbgs() << "Running secret-distribute-generic ";
      if (opsToDistribute.empty()) {
        llvm::dbgs() << "on all ops\n";
      } else {
        llvm::dbgs() << "on ops: \n";
        for (const auto &op : opsToDistribute) {
          llvm::dbgs() << " - " << op << "\n";
        }
      }
    });

    patterns.add<SplitGeneric>(context, opsToDistribute);
    // These patterns are shared with canonicalization
    patterns.add<CollapseSecretlessGeneric, RemoveUnusedGenericArgs,
                 RemoveNonSecretGenericArgs>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
