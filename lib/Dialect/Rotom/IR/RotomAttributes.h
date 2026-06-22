#ifndef LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_
#define LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_

#include <cstdint>

#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
// IWYU pragma: begin_keep
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "mlir/include/mlir/IR/OpImplementation.h"    // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Rotom/IR/RotomAttributes.h.inc"

namespace mlir::heir::rotom {

enum class LayoutPieceKind { Traversal, Replication, Gap };

struct LayoutData {
  int64_t n;
  int64_t ctPrefixLen;
  llvm::SmallVector<DimAttr> originalDims;
  llvm::SmallVector<DimAttr> traversalDims;
  llvm::SmallVector<DimAttr> replicationDims;
  llvm::SmallVector<DimAttr> gapDims;
  llvm::SmallVector<LayoutPieceKind> pieces;
  llvm::SmallVector<int64_t> pieceIndex;
  // Parallel to `pieces`: each traversal piece reads one mixed-radix digit of its
  // tensor index i -- digit = (i / pieceStride) mod pieceExtent -- so pieceStride
  // is the within-axis divisor and pieceExtent the digit's extent. The most-
  // significant piece of an axis (pieceStride * pieceExtent == the axis's full
  // extent) needs no modulus, since i / pieceStride is already < pieceExtent;
  // the emitter drops it there. A whole dim packed as one piece uses stride 1
  // and extent = the full dim (digit == i). Non-traversal pieces use stride 1
  // and their own extent.
  llvm::SmallVector<int64_t> pieceStride;
  llvm::SmallVector<int64_t> pieceExtent;
};

/// Preprocess a Rotom layout.
FailureOr<LayoutData> preprocessLayoutAttr(LayoutAttr attr);

/// Computes how many leading entries of `dims` (read left-to-right) fall on the
/// ciphertext axis for a ciphertext of `n` slots: the prefix that does not fit
/// into the remaining slot budget. Shared so attribute preprocessing and the
/// layout cost utilities derive the ct/slot split identically.
size_t inferCtPrefixLen(llvm::ArrayRef<DimAttr> dims, int64_t n);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_
