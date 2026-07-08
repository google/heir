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

struct LayoutPiece {
  DimAttr dim;
  LayoutPieceKind kind;
  // axisIndex, divBy, and modBy lower a traversal piece into its term of the
  // ISL relation. The emitter builds an address `[i0, i1, ...] -> [ct, slot]`
  // with one variable per axis (LayoutData::axes); each piece contributes a
  // term reading one mixed-radix digit of its axis's variable.
  //
  // axisIndex picks the variable: an index into LayoutData::axes, emitted as
  // `i{axisIndex}`.
  int64_t axisIndex = -1;
  // divBy and modBy pick which digit of that variable the piece reads, as
  // (i / divBy) mod modBy. divBy is the digit's place value: the piece's
  // stride when the axis is split across pieces, else 1. modBy is the digit's
  // extent or 0 to drop the modulus on the most-significant digit.
  int64_t divBy = 1;
  int64_t modBy = 0;
};

struct LayoutData {
  int64_t n;
  // Pieces [0, ctPrefixLen) are the ciphertext dimensions.
  // Pieces [ctPrefixLen, pieces.size()) are the slot dimensions.
  // The split is shown with the `|` separator.
  int64_t ctPrefixLen;
  // Logical tensor axes.
  llvm::SmallVector<DimAttr> axes;
  llvm::SmallVector<LayoutPiece> pieces;

  bool isCiphertextPiece(size_t p) const {
    return static_cast<int64_t>(p) < ctPrefixLen;
  }
};

/// Preprocess a Rotom layout.
FailureOr<LayoutData> preprocessLayoutAttr(LayoutAttr attr);

/// Canonicalizes raw layout pieces to the stored form. When the slot side
/// (the longest dims suffix fitting `n`) underfills the ciphertext, it inserts
/// the explicit front gap piece at the ct/slot boundary.
void canonicalizeLayoutDims(MLIRContext* ctx, llvm::SmallVector<DimAttr>& dims,
                            int64_t n, llvm::SmallVector<int64_t>& rolls);

/// Computes how many leading entries of `dims` (read left-to-right) fall on the
/// ciphertext axis for a ciphertext of `n` slots: the prefix that does not fit
/// into the remaining slot budget. Shared so attribute preprocessing and the
/// layout cost utilities derive the ct/slot split identically.
size_t inferCtPrefixLen(llvm::ArrayRef<DimAttr> dims, int64_t n);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_H_
