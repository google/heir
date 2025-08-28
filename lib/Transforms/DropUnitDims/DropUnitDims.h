#ifndef LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_H_
#define LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_H_

#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/DropUnitDims/DropUnitDims.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/DropUnitDims/DropUnitDims.h.inc"

// Returns a list of unit dims of a type
SmallVector<int64_t> getUnitDims(ShapedType type);

/// Returns a collapsed `val` where the collapsing occurs at dims in positions.
Value collapseDimsAt(PatternRewriter& rewriter, Value val,
                     ArrayRef<int64_t> positions);

/// Collapse all collapsible operands.
SmallVector<Value> collapseOperands(PatternRewriter& rewriter,
                                    ArrayRef<Value> operands,
                                    ArrayRef<int64_t> collapseDims);

/// Expand result tensor.
Value expandResult(PatternRewriter& rewriter, Value result,
                   RankedTensorType expandedType, SmallVector<int64_t> dims);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_H_
