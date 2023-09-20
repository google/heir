#include "include/Dialect/Secret/Transforms/DistributeGeneric.h"

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETDISTRIBUTEGENERIC
#include "include/Dialect/Secret/Transforms/Passes.h.inc"

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
      : OpRewritePattern<GenericOp>(context, /*benefit=*/2) {}

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
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    bool hasUnusedOps = false;
    Block *body = op.getBody();

    for (size_t i = 0; i < body->getArguments().size(); ++i) {
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

// Split the first op off from a multi-op secret.generic. If the op does not use
// any secret values, CollapseSecretlessGeneric will apply to remove the generic
// op entirely.
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
struct PeelFromGeneric : public OpRewritePattern<GenericOp> {
  PeelFromGeneric(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    Block *body = op.getBody();
    unsigned numOps = body->getOperations().size();
    if (numOps == 0) {
      llvm_unreachable(
          "secret.generic must have nonempty body (the verifier should enforce "
          "this)");
    }

    // Recursive base case: stop if there's only one op left, noting that we
    // check for 2 ops because the last op is enforced to be a yield op by the
    // verifier.
    if (numOps == 1 || numOps == 2) {
      return failure();
    }

    Operation &firstOp = body->getOperations().front();
    // The inputs to the op are generic op's block arguments (cleartext values),
    // and they need to change to be the corresponding generic op's normal
    // operands (maybe secret values).
    SmallVector<Value> newInputs;
    // The indices of the new inputs in the original block argument list
    SmallVector<unsigned> newInputIndices;
    for (Value val : firstOp.getOperands()) {
      int index = std::find(body->getArguments().begin(),
                            body->getArguments().end(), val) -
                  body->getArguments().begin();
      newInputIndices.push_back(index);
      newInputs.push_back(op.getOperand(index));
    }

    // Result types are secret versions of the results of the block's only op
    SmallVector<Type> newResultTypes;
    for (Type ty : firstOp.getResultTypes()) {
      newResultTypes.push_back(SecretType::get(ty));
    }

    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), newInputs, newResultTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
          auto *newOp = b.clone(firstOp);
          newOp->setOperands(blockArguments);
          b.create<YieldOp>(loc, newOp->getResults());
        });

    // Once the op is split off into a new generic op, we need to add new
    // operands to the old generic op, add new corresponding block arguments,
    // and replace all uses of the firstOp's results with the created block
    // arguments.
    SmallVector<Value> oldGenericNewBlockArgs;
    rewriter.updateRootInPlace(op, [&]() {
      op.getInputsMutable().append(newGeneric.getResults());
      for (auto ty : firstOp.getResultTypes()) {
        BlockArgument arg = op.getBody()->addArgument(ty, firstOp.getLoc());
        oldGenericNewBlockArgs.push_back(arg);
      }
    });
    rewriter.replaceOp(&firstOp, oldGenericNewBlockArgs);

    return success();
  }
};

struct DistributeGeneric
    : impl::SecretDistributeGenericBase<DistributeGeneric> {
  using SecretDistributeGenericBase::SecretDistributeGenericBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<PeelFromGeneric, CollapseSecretlessGeneric,
                 RemoveUnusedGenericArgs>(context);
    // TODO(https://github.com/google/heir/issues/170): add a pattern that
    // distributes generic through a single op containing one or more regions.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
