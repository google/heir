#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::rotom {
namespace {

/// Maps a tensor axis id to its iterator index `i*` after preprocessing.
/// Preprocessing dedupes traversal pieces per logical axis (a mixed-radix
/// split contributes one variable), so the axis id identifies the variable.
static FailureOr<int64_t> varIndexForAxis(const SmallVector<DimAttr>& axes,
                                          int64_t axis) {
  for (int64_t i = 0; i < static_cast<int64_t>(axes.size()); ++i) {
    if (axes[i].getDim() == axis) return i;
  }
  return failure();
}

/// Product of the extents of `dim`'s pieces in the layout's dims list (the
/// logical axis extent; a single whole-dim piece gives its own size).
static int64_t fullExtentOfDim(ArrayAttr rotomDims, int64_t dim) {
  int64_t extent = 1;
  for (Attribute a : rotomDims) {
    auto d = cast<DimAttr>(a);
    if (!d.isGap() && !d.isReplicate() && d.getDim() == dim) {
      extent *= d.getSize();
    }
  }
  return extent;
}

/// Whether `dim` is packed as more than one mixed-radix piece.
static bool isSplitDim(ArrayAttr rotomDims, int64_t dim) {
  int64_t count = 0;
  for (Attribute a : rotomDims) {
    auto d = cast<DimAttr>(a);
    if (!d.isGap() && !d.isReplicate() && d.getDim() == dim) ++count;
  }
  return count > 1;
}

static std::string modExpr(llvm::StringRef expr, int64_t mod) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "(" << expr << " - " << mod << " * floor((" << expr << ") / " << mod
     << "))";
  return out;
}

static std::string floorDivExpr(llvm::StringRef expr, int64_t d) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "floor((" << expr << ") / " << d << ")";
  return out;
}

/// Index of the gap piece at position `idx` of the layout's dims list within
/// the per-kind gap dim list (gaps are collected in order of appearance).
static int64_t gapIndexForPosition(ArrayAttr rotomDims, int64_t idx) {
  return llvm::count_if(
      rotomDims.getValue().take_front(idx),
      [](Attribute attr) { return cast<DimAttr>(attr).isGap(); });
}

/// Ordinal of pieces[p] among the pieces of its kind. Used to name the
/// existential variables for replication and gap pieces, and matches the
/// dims-list ordinals the roll machinery computes.
static int64_t kindOrdinal(ArrayRef<LayoutPiece> pieces, size_t p) {
  int64_t ordinal = 0;
  for (size_t q = 0; q < p; ++q) {
    if (pieces[q].kind == pieces[p].kind) ++ordinal;
  }
  return ordinal;
}

/// Gap dims referenced in the roll-by position, as indices into the gap dim
/// list. A rolled-by gap claims its blocks (each holds a distinct rotation of
/// the rolled dim), so it gets an existential variable instead of folding to
/// zero.
static llvm::DenseSet<int64_t> rolledGapIndexSet(ArrayRef<int64_t> rolls,
                                                 ArrayAttr rotomDims) {
  llvm::DenseSet<int64_t> rolled;
  if (!rotomDims) return rolled;
  for (size_t i = 0; i + 1 < rolls.size(); i += 2) {
    const int64_t toIdx = rolls[i + 1];
    if (toIdx < 0 || toIdx >= static_cast<int64_t>(rotomDims.size())) continue;
    if (cast<DimAttr>(rotomDims[toIdx]).isGap()) {
      rolled.insert(gapIndexForPosition(rotomDims, toIdx));
    }
  }
  return rolled;
}

static LogicalResult emitSegmentAddress(
    llvm::raw_ostream& os, bool& firstTerm, ArrayRef<LayoutPiece> pieces,
    const SmallVector<DimAttr>& axes, int64_t numAxes, size_t segStart,
    size_t segEnd, const llvm::DenseSet<int64_t>& rolledGapIndices,
    ArrayRef<int64_t> rolls, ArrayRef<int64_t> rollSteps, ArrayAttr rotomDims) {
  // Effective extent of a piece = the range of its mixed-radix digit, which
  // is the piece's own extent: a lone piece reads the whole axis index, and
  // for a valid mixed-radix split each digit ranges over its piece's extent.
  llvm::SmallVector<int64_t> suffixCoeff(pieces.size(), 0);
  int64_t suffix = 1;
  for (size_t p = segEnd; p > segStart;) {
    --p;
    suffixCoeff[p] = suffix;
    suffix *= pieces[p].dim.getSize();
  }

  auto emitTerm = [&](int64_t coeff, llvm::StringRef expr) -> LogicalResult {
    if (coeff == 0) return failure();
    if (firstTerm) {
      if (coeff < 0) os << "-";
      firstTerm = false;
    } else {
      os << (coeff < 0 ? " - " : " + ");
    }
    const int64_t absCoeff = std::llabs(coeff);
    if (absCoeff == 1) {
      os << expr;
    } else {
      os << absCoeff << " * " << expr;
    }
    return success();
  };

  llvm::DenseMap<int64_t, int64_t> gapCoeff;
  llvm::DenseMap<int64_t, int64_t> replicationCoeff;
  for (size_t p = segStart; p < segEnd; ++p) {
    const int64_t coeff = suffixCoeff[p];
    if (pieces[p].kind == LayoutPieceKind::Gap) {
      // Only a rolled-by gap contributes an address term (its blocks hold the
      // rotations of the rolled dim); other gaps fold to zero, leaving their
      // blocks unclaimed.
      const int64_t ordinal = kindOrdinal(pieces, p);
      if (!rolledGapIndices.contains(ordinal)) continue;
      gapCoeff[ordinal] = coeff;
    } else if (pieces[p].kind == LayoutPieceKind::Replication) {
      replicationCoeff[kindOrdinal(pieces, p)] = coeff;
    }
  }

  // A tensor dim can contribute several pieces to one segment (a mixed-radix
  // split places more than one digit of the same index on the same axis), so
  // collect a list of (coeff, digit descriptor) per dim rather than one entry.
  struct AxisDigit {
    int64_t coeff;
    int64_t divBy;
    int64_t modBy;
  };
  llvm::DenseMap<int64_t, llvm::SmallVector<AxisDigit>> digitsByAxis;
  for (size_t p = segStart; p < segEnd; ++p) {
    if (pieces[p].kind != LayoutPieceKind::Traversal) continue;
    const int64_t ti = pieces[p].axisIndex;
    if (axes[ti].getSize() == 1) continue;
    digitsByAxis[ti].push_back(
        {suffixCoeff[p], pieces[p].divBy, pieces[p].modBy});
  }

  llvm::SmallVector<std::string> axisExprs;
  axisExprs.reserve(axes.size());
  for (int64_t i = 0; i < static_cast<int64_t>(axes.size()); ++i) {
    axisExprs.push_back("i" + std::to_string(i));
  }

  // Apply rolls left-to-right. A piece FROM rewrites that piece's own
  // mixed-radix digit in place, (digit - step * shift) mod extent(piece),
  // leaving its axis's other digits untouched; an `axis` FROM rewrites the
  // whole axis index mod its full extent, so the shift borrows across digits
  // (each piece then takes its digit of the rolled index). The shift is the
  // by endpoint's index: a traversal piece's digit (its whole index when the
  // axis is unsplit), a whole axis via `axis`, a slot replication's replica
  // index, or a gap's block index. Layout rolls always have unit step; a
  // non-unit step arises only from a plan-level PreRotation appended by the
  // caller (the baby-step/giant-step giant pre-rotation), which multiplies
  // the shift and runs against the subtraction direction.
  //
  // The rewrite lands wherever the FROM pieces sit -- the ciphertext
  // address, the slot address, or both when the axis straddles the boundary.
  // A roll diagonalizes a ciphertext dimension against a slot one (one
  // ciphertext per diagonal) or two slot dimensions (a Halevi-Shoup slot
  // diagonal); a roll within the ciphertext axis is a free ciphertext
  // relabeling that enumeration never generates.
  if (!rolls.empty()) {
    if (!rotomDims || rolls.size() % 2 != 0) return failure();
    for (size_t i = 0; i < rolls.size(); i += 2) {
      const RollEndpoint from = decodeRollEndpoint(rolls[i]);
      const RollEndpoint by = decodeRollEndpoint(rolls[i + 1]);
      const int64_t step = i / 2 < rollSteps.size() ? rollSteps[i / 2] : 1;
      if (!from.isAxis && from.index >= static_cast<int64_t>(rotomDims.size()))
        return failure();
      if (!by.isAxis && by.index >= static_cast<int64_t>(rotomDims.size()))
        return failure();
      DimAttr fromPiece =
          from.isAxis ? DimAttr() : dyn_cast<DimAttr>(rotomDims[from.index]);
      if (!from.isAxis && !fromPiece) return failure();
      const int64_t fromAxis = from.isAxis ? from.index : fromPiece.getDim();
      FailureOr<int64_t> maybeFromVar = varIndexForAxis(axes, fromAxis);
      if (failed(maybeFromVar)) return failure();
      const int64_t fromVar = *maybeFromVar;

      std::string toExpr;
      if (by.isAxis) {
        // Rolling by a whole axis: the shift is the axis's (possibly
        // already-rolled) index expression.
        FailureOr<int64_t> maybeToVar = varIndexForAxis(axes, by.index);
        if (failed(maybeToVar)) return failure();
        toExpr = axisExprs[*maybeToVar];
      } else {
        auto toDim = dyn_cast<DimAttr>(rotomDims[by.index]);
        if (!toDim) return failure();
        if (toDim.isGap()) {
          // Rolling by a gap dim: the shift is the gap's existential block
          // index, so block g holds the rolled index cyclically shifted by g.
          toExpr =
              "g" + std::to_string(gapIndexForPosition(rotomDims, by.index));
        } else if (toDim.isReplicate()) {
          // Rolling by a replication dim: the shift is the replica index,
          // whose existential variable is named after the piece's replication
          // slot.
          int64_t replicationIndex = 0;
          for (int64_t k = 0; k < by.index; ++k) {
            if (cast<DimAttr>(rotomDims[k]).isReplicate()) ++replicationIndex;
          }
          toExpr = "d" + std::to_string(numAxes + replicationIndex);
        } else {
          FailureOr<int64_t> maybeToVar = varIndexForAxis(axes, toDim.getDim());
          if (failed(maybeToVar)) return failure();
          toExpr = axisExprs[*maybeToVar];
          if (isSplitDim(rotomDims, toDim.getDim())) {
            // The by piece's digit: (i / stride) mod extent, with the modulus
            // redundant on the most-significant digit.
            const int64_t fullTo = fullExtentOfDim(rotomDims, toDim.getDim());
            if (toDim.getStride() > 1) {
              toExpr = floorDivExpr(toExpr, toDim.getStride());
            }
            if (toDim.getStride() * toDim.getSize() < fullTo) {
              toExpr = modExpr(toExpr, toDim.getSize());
            }
          }
        }
      }

      if (step != 1) {
        toExpr = std::to_string(step) + " * (" + toExpr + ")";
      }

      if (from.isAxis || !isSplitDim(rotomDims, fromAxis)) {
        // Whole-axis rewrite (the piece spelling of an unsplit axis is the
        // same operation: its one piece's digit is the axis index).
        std::string diffExpr = "(" + axisExprs[fromVar] + " - " + toExpr + ")";
        axisExprs[fromVar] =
            modExpr(diffExpr, fullExtentOfDim(rotomDims, fromAxis));
      } else {
        // Piece rewrite on a split axis: replace this piece's digit with
        // (digit - step * shift) mod extent(piece), i.e. add
        // stride * (newDigit - digit) to the axis index. No borrow crosses
        // into the other digits.
        const int64_t stride = fromPiece.getStride();
        const int64_t extent = fromPiece.getSize();
        const std::string axisExpr = axisExprs[fromVar];
        std::string digit = axisExpr;
        if (stride > 1) digit = floorDivExpr(digit, stride);
        if (stride * extent < fullExtentOfDim(rotomDims, fromAxis)) {
          digit = modExpr(digit, extent);
        }
        const std::string newDigit =
            modExpr("(" + digit + " - " + toExpr + ")", extent);
        axisExprs[fromVar] = "(" + axisExpr + " + " + std::to_string(stride) +
                             " * (" + newDigit + ") - " +
                             std::to_string(stride) + " * (" + digit + "))";
      }
    }
  }

  for (int64_t oldIdx = 0; oldIdx < static_cast<int64_t>(axes.size());
       ++oldIdx) {
    if (axes[oldIdx].getSize() == 1) continue;
    auto it = digitsByAxis.find(oldIdx);
    if (it == digitsByAxis.end()) continue;
    for (const AxisDigit& tp : it->second) {
      // Mixed-radix digit: (i / divBy) mod modBy (modBy 0 => no modulus). A
      // whole-dim piece (divBy 1, modBy 0) leaves the index untouched.
      std::string expr = axisExprs[oldIdx];
      if (tp.divBy > 1) expr = floorDivExpr(expr, tp.divBy);
      if (tp.modBy > 0) expr = modExpr(expr, tp.modBy);
      if (failed(emitTerm(tp.coeff, expr))) return failure();
    }
  }
  for (int64_t g = 0; g < static_cast<int64_t>(pieces.size()); ++g) {
    auto it = gapCoeff.find(g);
    if (it != gapCoeff.end() &&
        failed(emitTerm(it->second, "g" + std::to_string(g))))
      return failure();
  }
  for (int64_t e = 0; e < static_cast<int64_t>(pieces.size()); ++e) {
    auto it = replicationCoeff.find(e);
    if (it != replicationCoeff.end()) {
      const auto varName = "d" + std::to_string(numAxes + e);
      if (failed(emitTerm(it->second, varName))) return failure();
    }
  }
  return success();
}

static FailureOr<std::string> emitSplitCtSlotIsl(
    int64_t n, size_t prefix, ArrayRef<LayoutPiece> pieces,
    const SmallVector<DimAttr>& axes, ArrayRef<int64_t> rolls,
    ArrayRef<int64_t> rollSteps, ArrayAttr rotomDims) {
  if (prefix > pieces.size()) return failure();

  const int64_t numAxes = static_cast<int64_t>(axes.size());
  int64_t numReplication = 0;
  for (const LayoutPiece& piece : pieces) {
    if (piece.kind == LayoutPieceKind::Replication) ++numReplication;
  }

  const llvm::DenseSet<int64_t> rolledGapIndices =
      rolledGapIndexSet(rolls, rotomDims);

  int64_t numCt = 1;
  for (size_t p = 0; p < prefix; ++p) {
    const LayoutPiece& piece = pieces[p];
    if (piece.kind == LayoutPieceKind::Traversal ||
        piece.kind == LayoutPieceKind::Replication) {
      numCt *= piece.dim.getSize();
    } else if (piece.kind == LayoutPieceKind::Gap &&
               rolledGapIndices.contains(kindOrdinal(pieces, p))) {
      // A rolled-by gap claims its blocks (one rotation per block index).
      numCt *= piece.dim.getSize();
    }
  }
  if (numCt < 1) numCt = 1;

  std::string s;
  llvm::raw_string_ostream os(s);

  os << "{ [";
  for (int64_t i = 0; i < numAxes; ++i) {
    if (i) os << ", ";
    os << "i" << i;
  }
  os << "] -> [ct, slot] : ";

  bool first = true;
  auto emitAnd = [&]() {
    if (!first) os << " and ";
    first = false;
  };

  for (int64_t i = 0; i < numAxes; ++i) {
    const DimAttr d = axes[i];
    emitAnd();
    os << "0 <= i" << i << " <= " << (d.getSize() - 1);
  }

  emitAnd();
  os << "0 <= ct <= " << (numCt - 1);
  emitAnd();
  os << "0 <= slot <= " << (n - 1);

  const int64_t numLocalVars =
      numReplication + static_cast<int64_t>(rolledGapIndices.size());

  if (numLocalVars > 0) {
    os << " and exists ";
    bool firstVar = true;
    for (size_t p = 0; p < pieces.size(); ++p) {
      if (pieces[p].kind == LayoutPieceKind::Replication) {
        if (!firstVar) os << ", ";
        firstVar = false;
        os << "d" << (numAxes + kindOrdinal(pieces, p));
      } else if (pieces[p].kind == LayoutPieceKind::Gap &&
                 rolledGapIndices.contains(kindOrdinal(pieces, p))) {
        if (!firstVar) os << ", ";
        firstVar = false;
        os << "g" << kindOrdinal(pieces, p);
      }
    }
    os << " : ";
    first = true;
  }

  emitAnd();
  bool firstTerm = true;
  os << "ct = ";
  if (failed(emitSegmentAddress(os, firstTerm, pieces, axes, numAxes, 0, prefix,
                                rolledGapIndices, rolls, rollSteps, rotomDims)))
    return failure();
  if (firstTerm) os << "0";

  emitAnd();
  firstTerm = true;
  os << "slot = ";
  if (failed(emitSegmentAddress(os, firstTerm, pieces, axes, numAxes, prefix,
                                pieces.size(), rolledGapIndices, rolls,
                                rollSteps, rotomDims)))
    return failure();
  if (firstTerm) os << "0";

  if (numLocalVars > 0) {
    for (size_t p = 0; p < pieces.size(); ++p) {
      if (pieces[p].kind != LayoutPieceKind::Replication) continue;
      emitAnd();
      os << "0 <= d" << (numAxes + kindOrdinal(pieces, p))
         << " <= " << (pieces[p].dim.getSize() - 1);
    }
    for (size_t p = 0; p < pieces.size(); ++p) {
      if (pieces[p].kind != LayoutPieceKind::Gap) continue;
      const int64_t g = kindOrdinal(pieces, p);
      if (!rolledGapIndices.contains(g)) continue;
      emitAnd();
      os << "0 <= g" << g << " <= " << (pieces[p].dim.getSize() - 1);
    }
  }

  os << " }";
  os.flush();
  return s;
}

static FailureOr<std::string> lowerToIslImpl(LayoutAttr layout,
                                             const PreRotation* preRotation) {
  auto maybeData = preprocessLayoutAttr(layout);
  if (failed(maybeData)) return failure();
  const LayoutData& data = *maybeData;

  llvm::DenseMap<int64_t, llvm::DenseSet<int64_t>> seenDimStride;
  for (DimAttr d : data.axes) {
    auto& seenStridesForDim = seenDimStride[d.getDim()];
    if (seenStridesForDim.contains(d.getStride())) return failure();
    seenStridesForDim.insert(d.getStride());
  }

  DenseI64ArrayAttr rollsAttr = layout.getRolls();
  SmallVector<int64_t> rolls;
  if (rollsAttr)
    rolls.assign(rollsAttr.asArrayRef().begin(), rollsAttr.asArrayRef().end());
  // A plan-level pre-rotation composes as one more rewrite AFTER the
  // layout's rolls, expressed in the emitter's internal shift arithmetic:
  // (idx - step * digit) mod extent with step = -multiplier.
  SmallVector<int64_t> rollSteps(rolls.size() / 2, 1);
  if (preRotation) {
    rolls.push_back(preRotation->fromPiece);
    rolls.push_back(preRotation->byPiece);
    rollSteps.push_back(-preRotation->multiplier);
  }
  return emitSplitCtSlotIsl(data.n, data.ctPrefixLen, data.pieces, data.axes,
                            rolls, rollSteps, layout.getDims());
}

}  // namespace

FailureOr<std::string> RotomTensorExtLayoutLowering::lowerToTensorExtIsl(
    LayoutAttr layout, const PreRotation& preRotation) {
  return lowerToIslImpl(layout, &preRotation);
}

FailureOr<std::string> RotomTensorExtLayoutLowering::lowerToTensorExtIsl(
    LayoutAttr layout) {
  return lowerToIslImpl(layout, /*preRotation=*/nullptr);
}

}  // namespace mlir::heir::rotom
