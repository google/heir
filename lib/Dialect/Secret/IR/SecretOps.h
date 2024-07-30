#ifndef LIB_DIALECT_SECRET_IR_SECRETOPS_H_
#define LIB_DIALECT_SECRET_IR_SECRETOPS_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

// don't clobber import order
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Secret/IR/SecretOps.h.inc"

namespace mlir {
namespace heir {
namespace secret {

// Extracts the given op from inside the generic body and lifting to a new
// single-op generic after the context generic op. This function assumes as a
// precondition that the opToExtract's results do not have any uses besides in
// the yield of the genericOp. The HoistOpAfterGeneric pattern tests for this
// precondition.
//
// Replaces `genericOp` with a new genericOp using `rewriter`, and returns
// the two newly created generic ops, with the first one being the replacement
// for the input `genericOp`, and the second one being the extracted genericOp.
//
// Handles adding the operands of opToExtract to the yielded values of the
// generic. The new yields may not be needed, and this can be cleaned up by
// canonicalize, or a manual application of DedupeYieldedValues and
// RemoveUnusedYieldedValues.
std::pair<GenericOp, GenericOp> extractOpAfterGeneric(
    GenericOp genericOp, Operation *opToExtract, PatternRewriter &rewriter);

void populateGenericCanonicalizers(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_SECRET_IR_SECRETOPS_H_
