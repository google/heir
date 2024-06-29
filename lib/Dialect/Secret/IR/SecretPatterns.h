#ifndef LIB_DIALECT_SECRET_IR_SECRETPATTERNS_H_
#define LIB_DIALECT_SECRET_IR_SECRETPATTERNS_H_

#include <utility>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
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

// Find two adjacent generic ops and merge them into one.
// Accepts a parent op to apply this pattern only to generics descending from
// that op.
struct MergeAdjacentGenerics : public OpRewritePattern<GenericOp> {
  MergeAdjacentGenerics(mlir::MLIRContext *context,
                        std::optional<Operation *> parentOp = std::nullopt)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1),
        parentOp(parentOp) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;

 private:
  std::optional<Operation *> parentOp;
};

// Find a memeref that is stored to in the body of the generic, but not
// yielded, and add it to the yielded values.
struct YieldStoredMemrefs : public OpRewritePattern<GenericOp> {
  YieldStoredMemrefs(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;
};

// Dedupe duplicate values yielded by a generic
struct DedupeYieldedValues : public OpRewritePattern<GenericOp> {
  DedupeYieldedValues(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;
};

// Hoist an op out of a generic and place it before the generic (in a new
// generic block), if possible. This will be impossible if one of the op's
// operands depends on another SSA value defined by an op inside the
// generic.
//
// Accepts a list of op names to hoist.
struct HoistOpBeforeGeneric : public OpRewritePattern<GenericOp> {
  HoistOpBeforeGeneric(mlir::MLIRContext *context,
                       std::vector<std::string> opTypes)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1),
        opTypes(std::move(opTypes)) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;

  bool canHoist(Operation &op, GenericOp genericOp) const;

 private:
  std::vector<std::string> opTypes;
};

// Hoist an op out of a generic and place it after the generic (in a new
// generic block), if possible. This will be impossible if one of the op's
// results is used by another op inside the generic before the yield.
//
// Accepts a list of op names to hoist.
struct HoistOpAfterGeneric : public OpRewritePattern<GenericOp> {
  HoistOpAfterGeneric(mlir::MLIRContext *context,
                      std::vector<std::string> opTypes)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1),
        opTypes(std::move(opTypes)) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;

  bool canHoist(Operation &op) const;

 private:
  std::vector<std::string> opTypes;
};

// Identify the earliest op inside a generic that relies only on plaintext
// operands, and hoist it out of the generic.
struct HoistPlaintextOps : public OpRewritePattern<GenericOp> {
  HoistPlaintextOps(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override;
};

// Inspects a generic body for any constant operands, and copies the constant
// definition inside the generic body. This is useful for performing IR
// optimizations local to the generic body, so that constants can be folded
// rather than treated as variable arguments defined outside of the block.
void genericAbsorbConstants(secret::GenericOp genericOp,
                            mlir::IRRewriter &rewriter);

// Absorbs any memref deallocations into the generic body.
void genericAbsorbDealloc(secret::GenericOp genericOp,
                          mlir::IRRewriter &rewriter);

// Extract the body of a secret.generic into a function and replace the generic
// body with a call to the created function.
LogicalResult extractGenericBody(secret::GenericOp genericOp,
                                 mlir::IRRewriter &rewriter);

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_SECRET_IR_SECRETPATTERNS_H_
