#ifndef LIB_DIALECT_UTILS_H_
#define LIB_DIALECT_UTILS_H_

#include <cstdint>

#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

namespace mlir {
namespace heir {

/// Given a tensor::InsertOp or tensor::ExtractOp, and assuming the shape
/// of the input tensor is 1-dimensional and the input index is constant,
/// return the constant index value. If any of these conditions are not
/// met, return a failure.
template <typename Op>
FailureOr<int64_t> get1DExtractionIndex(Op op) {
  auto insertIndices = op.getIndices();
  if (insertIndices.size() != 1) return failure();

  // Each index must be constant; this may require running --canonicalize or
  // -sccp before this pass to apply folding rules (use -sccp if you need to
  // fold constants through control flow).
  Value insertIndex = *insertIndices.begin();
  auto insertIndexConstOp = insertIndex.getDefiningOp<arith::ConstantIndexOp>();
  if (!insertIndexConstOp) return failure();

  auto insertOffsetAttr =
      llvm::dyn_cast<IntegerAttr>(insertIndexConstOp.getValue());
  if (!insertOffsetAttr) return failure();

  return insertOffsetAttr.getInt();
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_UTILS_H_
