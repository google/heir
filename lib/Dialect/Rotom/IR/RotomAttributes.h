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

// Per-piece role for a dimension that straddles the ciphertext/slot boundary
// (its low `straddleSlotExtent` values occupy slots, its high values index
// ciphertexts). The high piece sits in the ciphertext segment and the low piece
// in the slot segment; both reference the same traversal dim (one domain var).
enum class StraddleRole { None, High, Low };

struct LayoutData {
  int64_t n;
  int64_t ctPrefixLen;
  llvm::SmallVector<DimAttr> originalDims;
  llvm::SmallVector<DimAttr> traversalDims;
  llvm::SmallVector<DimAttr> replicationDims;
  llvm::SmallVector<DimAttr> gapDims;
  llvm::SmallVector<LayoutPieceKind> pieces;
  llvm::SmallVector<int64_t> pieceIndex;
  // Parallel to `pieces`: the straddle role of each piece (None for all pieces
  // unless a dimension spans the ct/slot boundary).
  llvm::SmallVector<StraddleRole> pieceStraddle;
  // The slot extent of the straddling dim (the low part); 0 if none straddles.
  int64_t straddleSlotExtent = 0;
};

/// Preprocess a Rotom layout.
FailureOr<LayoutData> preprocessLayoutAttr(LayoutAttr attr);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_
