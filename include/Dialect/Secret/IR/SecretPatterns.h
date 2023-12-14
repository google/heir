#ifndef INCLUDE_DIALECT_SECRET_IR_SECRETPATTERNS_H_
#define INCLUDE_DIALECT_SECRET_IR_SECRETPATTERNS_H_

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

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

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;
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
      : OpRewritePattern<GenericOp>(context, /*benefit=*/2) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;
};

// Remove unused yields of a secret.generic op
//
// E.g.,
//
//    %res0, %res1 = secret.generic
//       {
//     ^bb0(%used: i32, %unused: i32):
//       %0 = arith.constant 1 : i32
//       %1 = arith.constant 1 : i32
//       secret.yield %0, %1 : i32, i32
//    } -> (!secret.secret<i32>, !secret.secret<i32>)
//    ... <only use %res0> ...
//
// is transformed to
//
//    %res0 = secret.generic
//       ins(%value_sec : !secret.secret<i32>) {
//     ^bb0(%used: i32):
//       %0 = arith.constant 1 : i32
//       %1 = arith.constant 1 : i32
//       secret.yield %0, : i32
//    } -> (!secret.secret<i32>)
//
// The dead code elimination pass then removes any subsequent unused ops inside
// the generic.
struct RemoveUnusedYieldedValues : public OpRewritePattern<GenericOp> {
  RemoveUnusedYieldedValues(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/2) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;
};

// Remove non-secret args of a secret.generic op, since they can be referenced
// directly in the enclosing scope.
struct RemoveNonSecretGenericArgs : public OpRewritePattern<GenericOp> {
  RemoveNonSecretGenericArgs(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/2) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;
};

// Find a value used in the generic but only defined in the ambient scope (i.e.,
// not passed through the generic's input) and add it to the generic's operands
// and block arguments.
//
// This is the opposite of RemoveNonSecretGenericArgs, and is useful in the case
// of a pass that needs to have easy access to all the values used in the
// secret.generic body, such as YosysOptimizer.
struct CaptureAmbientScope : public OpRewritePattern<GenericOp> {
  CaptureAmbientScope(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_SECRET_IR_SECRETPATTERNS_H_
