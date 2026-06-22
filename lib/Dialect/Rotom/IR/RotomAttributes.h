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
  // Parallel to `pieces`: the mixed-radix digit each traversal piece reads from
  // its tensor index i -- digit = (i / pieceDivBy) mod pieceModBy, where
  // pieceModBy == 0 means "no modulus". A whole dim packed as one piece uses
  // (1, 0) => digit == i. A dim split across the ct/slot boundary becomes two
  // same-dim pieces: the ct (high) piece (L, 0) => i / L and the slot (low)
  // piece (1, L) => i mod L, where L is the slot-side extent. Non-traversal
  // pieces use (1, 0).
  llvm::SmallVector<int64_t> pieceDivBy;
  llvm::SmallVector<int64_t> pieceModBy;
};

/// Preprocess a Rotom layout.
FailureOr<LayoutData> preprocessLayoutAttr(LayoutAttr attr);

/// Result of `inferCtPrefixLen`: how many leading dims fall on the ciphertext
/// axis (`length`), and the slot-side extent of a boundary dim that straddles
/// the ct/slot split (`straddleSlotExtent`, 0 when none straddles).
struct CtPrefix {
  size_t length;
  int64_t straddleSlotExtent;
};

/// Computes how many leading entries of `dims` (read left-to-right) fall on the
/// ciphertext axis for a ciphertext of `n` slots. When the boundary dim spans
/// the ct/slot split, the returned `straddleSlotExtent` is the slot-side extent
/// it contributes -- its high `size / straddleSlotExtent` part indexes
/// ciphertexts -- and is 0 otherwise. Shared so attribute preprocessing and the
/// layout cost utilities derive the ct/slot split identically.
CtPrefix inferCtPrefixLen(llvm::ArrayRef<DimAttr> dims, int64_t n);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_
