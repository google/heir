#include "include/Dialect/Secret/Transforms/DistributeGeneric.h"

#include <algorithm>
#include <utility>

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
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

// Inline the inner block of a secret.generic that has no secret operands.
//
// E.g.,
//
//    %res = secret.generic ins(%value : i32) {
//     ^bb0(%clear_value: i32):
//       %c7 = arith.constant 7 : i32
//       %0 = arith.muli %clear_value, %c7 : i32
//       secret.yield %0 : i32
//    } -> (!secret.secret<i32>)
//
// is transformed to
//
//    %0 = arith.constant 0 : i32
//    %res = arith.muli %value, %0 : i32
//
struct CollapseSecretlessGeneric : public OpRewritePattern<GenericOp> {
  CollapseSecretlessGeneric(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
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
  }
};

// FIXME: move this pattern so it can be reused in canonicalization.
// Remove unused args of a secret.generic op
//
// E.g.,
//
//    %res = secret.generic
//       ins(%value_sec, %unused_sec : !secret.secret<i32>, !secret.secret<i32>)
//       {
//     ^bb0(%used: i32, %unused: i32):
//       %0 = arith.muli %used, %used : i32
//       secret.yield %0 : i32
//    } -> (!secret.secret<i32>)
//
// is transformed to
//
//    %res = secret.generic
//       ins(%value_sec : !secret.secret<i32>) {
//     ^bb0(%used: i32):
//       %0 = arith.muli %used, %used : i32
//       secret.yield %0 : i32
//    } -> (!secret.secret<i32>)
//
struct RemoveUnusedGenericArgs : public OpRewritePattern<GenericOp> {
  RemoveUnusedGenericArgs(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
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
};

// FIXME: move this pattern so it can be reused in canonicalization.
// Remove non-secret args of a secret.generic op, since they can be referenced
// directly in the enclosing scope.
struct RemoveNonSecretGenericArgs : public OpRewritePattern<GenericOp> {
  RemoveNonSecretGenericArgs(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
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

      // Before moving the loop out of the generic, connect the loop's operands
      // to the corresponding secret operands (via the block argument number).
      rewriter.startRootUpdate(genericOp);

      // Set the loop op's operands that came from the secret generic block
      // to be the the corresponding operand of the generic op.
      for (OpOperand &operand : opToDistribute.getOpOperands()) {
        BlockArgument blockArg = dyn_cast<BlockArgument>(operand.get());
        if (blockArg && blockArg.getOwner() == genericOp.getBody()) {
          BlockArgument initArg = cast<BlockArgument>(operand.get());
          operand.set(genericOp.getOperand(initArg.getArgNumber()));
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
      auto yieldedValues = loop.getYieldedValues();
      genericOp.getBody(0)->getTerminator()->setOperands(yieldedValues);
      // An affine.for op might not have a yielded value, and only manipulate
      // memrefs in its body. In this case, both the secret.generic and the
      // affine.for will yield nothing.
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
        op->moveBefore(genericOp.getBody(0)->getTerminator());
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
      Block *genericBlock = genericOp.getBody(0);
      genericBlock->walk([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (operand.getParentBlock() != genericBlock) {
            if (isa<SecretType>(operand.getType())) {
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

  void splitGenericAroundOp(GenericOp op, Operation &opToDistribute,
                            PatternRewriter &rewriter) const {
    assert(false && "not implemented");
  }

  void splitGenericAfterOp(GenericOp op, Operation &opToDistribute,
                           PatternRewriter &rewriter) const {
    LLVM_DEBUG({
      opToDistribute.emitRemark() << " splitting generic after this op\n";
    });
    Block *body = op.getBody();
    // The inputs to the op may be in the generic op's block arguments
    // (cleartext values with corresponding secrets), and they need to change
    // to be the corresponding generic op's normal operands (maybe secret
    // values).
    SmallVector<Value> newInputs;
    for (Value val : opToDistribute.getOperands()) {
      int index = std::find(body->getArguments().begin(),
                            body->getArguments().end(), val) -
                  body->getArguments().begin();
      if (index == body->getArguments().size()) {
        // The operand is not a block argument, so it must be referencing the
        // ambient scope.
        newInputs.push_back(val);
      } else {
        newInputs.push_back(op.getOperand(index));
      }
    }

    // Result types are secret versions of the results of the op, since the
    // secret will yield all of this op's results immediately.
    SmallVector<Type> newResultTypes;
    for (Type ty : opToDistribute.getResultTypes()) {
      newResultTypes.push_back(SecretType::get(ty));
    }

    // We may be adding non-secret inputs to the generic, but later patterns
    // will clean this up.
    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), newInputs, newResultTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
          auto *newOp = b.clone(opToDistribute);
          newOp->setOperands(blockArguments);
          b.create<YieldOp>(loc, newOp->getResults());
        });

    // Once the op is split off into a new generic op, we need to add new
    // operands to the old generic op, add new corresponding block arguments,
    // and replace all uses of the opToDistribute's results with the created
    // block arguments.
    SmallVector<Value> oldGenericNewBlockArgs;
    rewriter.updateRootInPlace(op, [&]() {
      op.getInputsMutable().append(newGeneric.getResults());
      for (auto ty : opToDistribute.getResultTypes()) {
        BlockArgument arg =
            op.getBody()->addArgument(ty, opToDistribute.getLoc());
        oldGenericNewBlockArgs.push_back(arg);
      }
    });
    rewriter.replaceOp(&opToDistribute, oldGenericNewBlockArgs);
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
      distributeThroughRegionHoldingOp(op, *opToDistribute, rewriter);
      return success();
    }

    if (first) {
      splitGenericAfterOp(op, *opToDistribute, rewriter);
    } else {
      splitGenericAroundOp(op, *opToDistribute, rewriter);
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
    patterns.add<CollapseSecretlessGeneric, RemoveUnusedGenericArgs>(context);
    patterns.add<RemoveNonSecretGenericArgs>(context);
    // TODO(https://github.com/google/heir/issues/170): add a pattern that
    // distributes generic through a single op containing one or more regions.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
