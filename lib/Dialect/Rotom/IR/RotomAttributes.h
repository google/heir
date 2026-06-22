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
  // pieceDivBy and pieceModBy are a piece's Rotom stride and extent (from
  // [dim:extent:stride]), named for how the emitter consumes them: the mixed-
  // radix digit of tensor index i is (i / pieceDivBy) mod pieceModBy. pieceModBy
  // is 0 on the most-significant digit of an axis -- there i / pieceDivBy is
  // already below the extent, so the modulus is redundant and dropped. A whole
  // dim packed as one piece uses (1, 0) => digit == i; non-traversal pieces use
  // (1, 0).
  llvm::SmallVector<int64_t> pieceDivBy;
  llvm::SmallVector<int64_t> pieceModBy;
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
