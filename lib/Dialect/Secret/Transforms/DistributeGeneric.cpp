#include "lib/Dialect/Secret/Transforms/DistributeGeneric.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <utility>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"             // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "distribute-generic"

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETDISTRIBUTEGENERIC
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

std::optional<Value> ofrToValue(std::optional<OpFoldResult> ofr) {
  if (ofr.has_value()) {
    if (auto value = llvm::dyn_cast_if_present<Value>(*ofr)) {
      return value;
    }
  }
  return std::nullopt;
}

struct FoldSecretSeparators : public OpRewritePattern<GenericOp> {
  FoldSecretSeparators(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/4) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Erase a generic that only contains a separator operation and no results.
    auto &operations = op.getBody()->getOperations();
    if (operations.size() != 2 ||
        !isa<secret::SeparatorOp>(operations.front())) {
      return failure();
    }

    if (op.getNumResults() > 0) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

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

      rewriter.startOpModification(genericOp);

      LLVM_DEBUG(genericOp.emitRemark()
                 << "Generic op at start of distributeThroughRegionHoldingOp");

      // Clone the loop to occur just before the generic containing it. This
      // will create a temporary dominance invalidity because, e.g., an arith
      // op that operates on cleartext i32s in the loop body will now occur
      // before that block argument is defined. We will patch this up once we
      // move these ops inside a generic in the body of the loop.
      //
      // We also need to ensure the operands to the loop that are generic block
      // arguments are converted to the corresponding secret input
      rewriter.setInsertionPoint(genericOp);
      IRMapping mp;
      for (Value iterInit : loop.getInits()) {
        if (auto *genericOperand =
                genericOp.getOpOperandForBlockArgument(iterInit)) {
          mp.map(iterInit, genericOperand->get());
        }
      }
      LoopLikeOpInterface clonedLoop =
          dyn_cast<LoopLikeOpInterface>(rewriter.clone(opToDistribute, mp));
      for (auto [operand, blockArg] :
           llvm::zip(clonedLoop.getInits(), clonedLoop.getRegionIterArgs())) {
        if (isa<SecretType>(operand.getType()))
          blockArg.setType(operand.getType());
      }

      LLVM_DEBUG(genericOp->getParentOp()->emitRemark()
                 << "after cloning loop before generic "
                    "(expected type conflicts here)");

      Block &clonedLoopBody = clonedLoop->getRegion(0).getBlocks().front();
      rewriter.setInsertionPoint(&clonedLoopBody.getOperations().front());

      // The secret generic is replacing the loop body, which means its outputs
      // must correspond exactly to the loop's yielded values.
      //
      // It is possible that the yield was yielding, say, a memref that the
      // loop body modified. In this case, we need to trace that back to a
      // generic operand and replace future uses of it with the corresponding
      // input value.
      SmallVector<Type> loopResultTypes = llvm::to_vector<4>(
          llvm::map_range(clonedLoop->getResults().getTypes(),
                          [](Type t) -> Type { return SecretType::get(t); }));

      // If the original loop's iter arg was a generic block argument, then the
      // new generic should take as input the loop's corresponding iter arg.
      SmallVector<Value> newGenericOperands;
      for (OpOperand &oldGenericOperand : genericOp->getOpOperands()) {
        auto blockArg = genericOp.getBody()->getArgument(
            oldGenericOperand.getOperandNumber());
        auto index = std::find(loop.getInits().begin(), loop.getInits().end(),
                               blockArg) -
                     loop.getInits().begin();
        if (index < loop.getInits().size()) {
          newGenericOperands.push_back(clonedLoop.getRegionIterArgs()[index]);
          continue;
        }

        newGenericOperands.push_back(oldGenericOperand.get());
      }

      SmallVector<Operation *> opsToErase;
      GenericOp newGenericOp = rewriter.create<GenericOp>(
          clonedLoopBody.getOperations().front().getLoc(), newGenericOperands,
          loopResultTypes,
          [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
            IRMapping mp;
            for (BlockArgument blockArg : genericOp.getBody()->getArguments()) {
              mp.map(blockArg, blockArguments[blockArg.getArgNumber()]);
            }
            int index = 0;
            for (Value value : newGenericOperands) {
              mp.map(value, blockArguments[index++]);
            }
            for (Operation &op : clonedLoopBody.getOperations()) {
              if (&op == &opToDistribute) continue;

              // This ensures that the loop terminator isn't added to the body
              // of the generic, and that it is not erased.
              if (op.hasTrait<OpTrait::IsTerminator>()) continue;

              LLVM_DEBUG(llvm::dbgs() << "Cloning " << op.getName() << "\n");
              b.clone(op, mp);
              opsToErase.push_back(&op);
            }

            SmallVector<Value, 4> newYieldedValues;
            for (Value oldYieldedValue :
                 clonedLoopBody.getTerminator()->getOperands()) {
              newYieldedValues.push_back(mp.lookup(oldYieldedValue));
            }
            b.create<YieldOp>(loc, newYieldedValues);
          });

      LLVM_DEBUG(genericOp->getParentOp()->emitRemark()
                 << "after creating new generic inside loop");

      // Handle the new loop terminator, which, if it used to yield the
      // plaintext value now yielded by the generic, now needs to yield the
      // result of the secret generic.

      Operation *clonedLoopTerminator = clonedLoopBody.getTerminator();
      // This should not introduce a type conflict because we ensured that the
      // generic yielded the cleartext analogue of what the original terminator
      // yielded.
      clonedLoopTerminator->setOperands(newGenericOp.getResults());
      auto resultValues = clonedLoop.getLoopResults();
      if (resultValues.has_value()) {
        for (auto [yieldType, resultVal] :
             llvm::zip(clonedLoopTerminator->getOperandTypes(),
                       resultValues.value())) {
          resultVal.setType(yieldType);
        }
      }

      LLVM_DEBUG(genericOp->getParentOp()->emitRemark()
                 << "after updating cloned loop yield op");

      rewriter.finalizeOpModification(genericOp);

      // To replace the original secret.generic, we need to find a suitable
      // replacement for any of its result values. There are two cases:
      //
      // 1. The generic yielded result values from the contained loop, in which
      // case the loop now yields the secret generic's result. In this case, we
      // can replace with the loop's results.
      //
      // 2. The generic yielded values modified in the affine loop scope, in
      // which case we need to find the original secret operand and replace
      // with that.
      if (loop.getLoopResults() == genericOp.getYieldOp().getValues()) {
        // Case 1:
        rewriter.replaceOp(genericOp, clonedLoop);
      } else {
        // Case 2:
        SmallVector<Value> replacements;
        for (Value value : genericOp.getYieldOp().getValues()) {
          assert(isa<BlockArgument>(value) &&
                 "not sure what to do here, file a bug");
          replacements.push_back(
              genericOp.getOperand(cast<BlockArgument>(value).getArgNumber()));
          LLVM_DEBUG(llvm::dbgs() << "replacing value " << value << " with "
                                  << replacements.back() << "\n");
        }
        rewriter.replaceOp(genericOp, replacements);
      }

      LLVM_DEBUG(clonedLoop->getParentOp()->emitRemark()
                 << "after replacing original generic with appropriate values");

      for (Operation *op : reverse(opsToErase)) {
        rewriter.eraseOp(op);
      }

      LLVM_DEBUG(clonedLoop->getParentOp()->emitRemark()
                 << "after erasing loop ops cloned into new generic");

      return;
    }

    // TODO(#307): handle RegionBranchOpInterface (scf.while, scf.if).
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
    mlir::DominanceInfo dom(sourceGeneric->getParentOfType<func::FuncOp>());
    opToMove.walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.get().getParentBlock() == sourceGeneric.getBody()) {
          LLVM_DEBUG(llvm::dbgs() << "opToMove depends on block argument "
                                  << operand.get() << " of source generic\n");
          assert(mlir::isa<BlockArgument>(operand.get()) &&
                 "opToMove has a non-block-argument operand defined in the "
                 "source generic");

          BlockArgument blockArg = cast<BlockArgument>(operand.get());
          Value sourceGenericArg =
              sourceGeneric.getOperand(blockArg.getArgNumber());
          LLVM_DEBUG(opToMove.emitRemark()
                     << "Moving op requires adding " << sourceGenericArg
                     << " to target generic\n");

          // The operand may be the result of the generic we'd like to move it
          // to, in which case the targetGeneric's result corresponds to a
          // yielded value in targetGeneric, and we can map the op's operand to
          // that yielded value.
          auto *definingOp = sourceGenericArg.getDefiningOp();
          if (definingOp == targetGeneric.getOperation()) {
            int resultIndex =
                std::find(targetGeneric.getResults().begin(),
                          targetGeneric.getResults().end(), sourceGenericArg) -
                targetGeneric.getResults().begin();

            assert(resultIndex >= 0 && "unable to find result in yield");
            Value yieldedValue =
                targetGeneric.getYieldOp()->getOperand(resultIndex);
            LLVM_DEBUG(llvm::dbgs() << "Mapping " << operand.get()
                                    << " to existing yielded value "
                                    << yieldedValue << "\n");
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

        // The op operand may be defined within a region contained in opToMove
        // so no need to map it's IR value.
        if (opToMove.getNumRegions() > 0 &&
            llvm::any_of(opToMove.getRegions(), [&](Region &region) {
              return region.isAncestor(operand.get().getParentRegion());
            })) {
          continue;
        }

        // If the operand is not a sourceGeneric block argument or defined
        // within one of its regions, then it must be a value defined in the
        // enclosing scope. It cannot be a value defined between the two
        // secret.generics, because in this pattern we only invoke this just
        // after splitting a generic into two adjacent ops.
        assert(mlir::isa<BlockArgument>(operand.get()) ||
               dom.properlyDominates(operand.get().getDefiningOp(),
                                     targetGeneric) &&
                   "Invalid use of moveOpToEarlierGeneric");
      }
    });

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
    auto newGeneric = genericOp.extractOpBeforeGeneric(&firstOp, rewriter);
    LLVM_DEBUG(newGeneric.emitRemark() << " created new generic op\n");
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
    patterns.add<FoldSecretSeparators, CollapseSecretlessGeneric,
                 RemoveUnusedGenericArgs, RemoveNonSecretGenericArgs>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
