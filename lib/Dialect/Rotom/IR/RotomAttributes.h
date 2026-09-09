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

// The dim id of a `K` piece; see DimAttr::isContraction.
inline constexpr int64_t kDimContraction = -3;

// One argument of a roll: either one piece of the layout (a position in its
// dims list) or a whole tensor axis (written `axis N`; legal only when the
// axis is packed as more than one piece).
struct RollArg {
  bool isAxis;
  int64_t index;  // Dims-list position, or the tensor axis id when isAxis.
  bool operator==(const RollArg& other) const {
    return isAxis == other.isAxis && index == other.index;
  }
};

// Argument encoding in the flat rolls storage: a piece argument is its
// non-negative dims-list position, an axis argument is -(axis + 1).
inline int64_t encodeRollArg(RollArg e) {
  return e.isAxis ? -(e.index + 1) : e.index;
}
inline RollArg decodeRollArg(int64_t encoded) {
  return encoded < 0 ? RollArg{true, -encoded - 1} : RollArg{false, encoded};
}

// One roll of a layout: FROM's index is rewritten to
// (idx_from - shift(by)) mod extent(from), where a piece FROM rewrites only
// the part of the axis index that piece reads, and an axis FROM rewrites the
// whole axis index.
struct RollSpec {
  RollArg from;
  RollArg by;
};

// The layout's rolls with both arguments decoded.
llvm::SmallVector<RollSpec> getRollSpecs(LayoutAttr layout);

// The traversal pieces one tensor axis occupies in a dims list: how many
// there are, and the product of their extents -- the axis's full logical
// extent, since a split factors it across pieces. Gap and
// replication pieces name no tensor axis and are skipped.
struct AxisPieces {
  int64_t count = 0;
  int64_t extent = 1;
  // Whether a roll must name this axis with an `axis` argument: with a single
  // piece, rolling the axis and rolling that piece are the same rewrite, and
  // the piece form is canonical.
  bool isSplit() const { return count > 1; }
};

AxisPieces axisPieces(llvm::ArrayRef<DimAttr> dims, int64_t axis);
AxisPieces axisPieces(ArrayAttr dims, int64_t axis);

enum class LayoutPieceKind { Traversal, Replication, Gap };

struct LayoutPiece {
  DimAttr dim;
  LayoutPieceKind kind;
  // axisIndex, divBy, and modBy lower a traversal piece into its term of the
  // ISL relation. The emitter builds an address `[i0, i1, ...] -> [ct, slot]`
  // with one variable per axis (LayoutData::axes); each piece contributes a
  // term reading one part of its axis's variable.
  //
  // axisIndex picks the variable: an index into LayoutData::axes, emitted as
  // `i{axisIndex}`.
  int64_t axisIndex = -1;
  // divBy and modBy pick which part of that variable the piece reads, as
  // (i / divBy) mod modBy. divBy is that part's offset: the piece's
  // stride when the axis is split across pieces, else 1. modBy is the piece's
  // extent, or 0 to drop the modulus on the highest part.
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
