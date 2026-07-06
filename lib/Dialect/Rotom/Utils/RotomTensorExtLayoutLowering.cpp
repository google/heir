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

/// Maps a `#rotom.dim` from the layout's `dims` list to its iterator index
/// `i*` after preprocessing. Preprocessing dedupes traversal dims per logical
/// axis (a mixed-radix split contributes one variable), so the piece's dim id
/// identifies the variable.
static FailureOr<int64_t> traversalIndexForRotomDim(
    const SmallVector<DimAttr>& traversalDims, DimAttr want) {
  for (int64_t i = 0; i < static_cast<int64_t>(traversalDims.size()); ++i) {
    if (traversalDims[i].getDim() == want.getDim()) return i;
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
  int64_t gapIndex = 0;
  for (int64_t k = 0; k < idx; ++k) {
    if (cast<DimAttr>(rotomDims[k]).isGap()) ++gapIndex;
  }
  return gapIndex;
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
    auto toDim = dyn_cast<DimAttr>(rotomDims[toIdx]);
    if (toDim && toDim.isGap()) {
      rolled.insert(gapIndexForPosition(rotomDims, toIdx));
    }
  }
  return rolled;
}

static LogicalResult emitSegmentAddress(
    llvm::raw_ostream& os, bool& firstTerm, ArrayRef<LayoutPieceKind> pieces,
    ArrayRef<int64_t> pieceIndex, ArrayRef<int64_t> pieceDivBy,
    ArrayRef<int64_t> pieceModBy, const SmallVector<DimAttr>& traversalDims,
    const SmallVector<DimAttr>& gapDims,
    const SmallVector<DimAttr>& replicationDims,
    int64_t numActiveTraversalComponents, size_t segStart, size_t segEnd,
    const llvm::DenseSet<int64_t>& rolledGapIndices, ArrayRef<int64_t> rolls,
    ArrayRef<int64_t> rollScales, ArrayAttr rotomDims) {
  // Effective extent of a piece = the range of its mixed-radix digit: modBy
  // when the piece reads i mod L (the slot/low part), else sz/divBy (the whole
  // dim when divBy 1, or the ct/high part sz/L of a straddling dim).
  auto pieceSize = [&](size_t p) -> int64_t {
    switch (pieces[p]) {
      case LayoutPieceKind::Traversal: {
        int64_t sz = traversalDims[pieceIndex[p]].getSize();
        if (pieceModBy[p] > 0) return pieceModBy[p];
        return sz / pieceDivBy[p];
      }
      case LayoutPieceKind::Gap:
        return gapDims[pieceIndex[p]].getSize();
      case LayoutPieceKind::Replication:
        return replicationDims[pieceIndex[p]].getSize();
    }
    return 1;
  };

  llvm::SmallVector<int64_t> suffixCoeff(pieces.size(), 0);
  int64_t suffix = 1;
  for (size_t p = segEnd; p > segStart;) {
    --p;
    suffixCoeff[p] = suffix;
    suffix *= pieceSize(p);
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
    if (pieces[p] == LayoutPieceKind::Gap) {
      // Only a rolled-by gap contributes an address term (its blocks hold the
      // rotations of the rolled dim); other gaps fold to zero, leaving their
      // blocks unclaimed.
      if (!rolledGapIndices.contains(pieceIndex[p])) continue;
      gapCoeff[pieceIndex[p]] = coeff;
    } else if (pieces[p] == LayoutPieceKind::Replication) {
      replicationCoeff[pieceIndex[p]] = coeff;
    }
  }

  // A tensor dim can contribute several pieces to one segment (a mixed-radix
  // split places more than one digit of the same index on the same axis), so
  // collect a list of (coeff, digit descriptor) per dim rather than one entry.
  struct TraversalPiece {
    int64_t coeff;
    int64_t divBy;
    int64_t modBy;
  };
  llvm::DenseMap<int64_t, llvm::SmallVector<TraversalPiece>> traversalPieces;
  for (size_t p = segStart; p < segEnd; ++p) {
    if (pieces[p] != LayoutPieceKind::Traversal) continue;
    const int64_t ti = pieceIndex[p];
    if (traversalDims[ti].getSize() == 1) continue;
    traversalPieces[ti].push_back(
        {suffixCoeff[p], pieceDivBy[p], pieceModBy[p]});
  }

  llvm::SmallVector<std::string> traversalExprs;
  traversalExprs.reserve(traversalDims.size());
  for (int64_t i = 0; i < static_cast<int64_t>(traversalDims.size()); ++i) {
    traversalExprs.push_back("i" + std::to_string(i));
  }

  // Apply roll(a,b) transforms left-to-right:
  // t_a <- (t_a - scale * t_b) mod fullExtent(dim(a)).
  //
  // Rolls are applied on both the ciphertext-address line and the slot line.
  // A roll only affects the address whose segment actually contains the rolled
  // `from` dim (the other line's coefficient for that dim is zero, so the
  // transformed expression is dropped by emitTerm). Applying on the ct line is
  // what lets a roll place a diagonal on the ciphertext axis (one ciphertext
  // per diagonal); restricting to the slot line would silently no-op such
  // ct-side rolls.
  //
  // A from piece of a mixed-radix split dim rewrites the WHOLE dim's index
  // (each piece then takes its digit of the rolled index); a by piece of a
  // split dim shifts by that piece's digit of the dim's current (possibly
  // already-rolled) expression. With a scale this expresses
  // baby-step/giant-step diagonal packings.
  if (!rolls.empty()) {
    if (!rotomDims || rolls.size() % 2 != 0) return failure();
    for (size_t i = 0; i < rolls.size(); i += 2) {
      const int64_t fromIdx = rolls[i];
      const int64_t toIdx = rolls[i + 1];
      const int64_t scale = i / 2 < rollScales.size() ? rollScales[i / 2] : 1;
      if (fromIdx < 0 || toIdx < 0 ||
          fromIdx >= static_cast<int64_t>(rotomDims.size()) ||
          toIdx >= static_cast<int64_t>(rotomDims.size())) {
        return failure();
      }
      auto fromDim = dyn_cast<DimAttr>(rotomDims[fromIdx]);
      auto toDim = dyn_cast<DimAttr>(rotomDims[toIdx]);
      if (!fromDim || !toDim) return failure();
      FailureOr<int64_t> maybeFromTrav =
          traversalIndexForRotomDim(traversalDims, fromDim);
      if (failed(maybeFromTrav)) return failure();
      const int64_t fromTrav = *maybeFromTrav;
      std::string toExpr;
      if (toDim.isGap()) {
        // Rolling by a gap dim: the shift is the gap's existential block
        // index, so block g holds the rolled dim cyclically shifted by g.
        toExpr = "g" + std::to_string(gapIndexForPosition(rotomDims, toIdx));
      } else if (toDim.isReplicate()) {
        // Rolling by a replication dim: the shift is the replica index, whose
        // existential variable is named after the piece's replication slot.
        int64_t replicationIndex = 0;
        for (int64_t k = 0; k < toIdx; ++k) {
          if (cast<DimAttr>(rotomDims[k]).isReplicate()) ++replicationIndex;
        }
        toExpr = "d" + std::to_string(numActiveTraversalComponents +
                                      replicationIndex);
      } else {
        FailureOr<int64_t> maybeToTrav =
            traversalIndexForRotomDim(traversalDims, toDim);
        if (failed(maybeToTrav)) return failure();
        toExpr = traversalExprs[*maybeToTrav];
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
      if (scale != 1) {
        toExpr = std::to_string(scale) + " * (" + toExpr + ")";
      }
      std::string diffExpr =
          "(" + traversalExprs[fromTrav] + " - " + toExpr + ")";
      traversalExprs[fromTrav] =
          modExpr(diffExpr, fullExtentOfDim(rotomDims, fromDim.getDim()));
    }
  }

  for (int64_t oldIdx = 0; oldIdx < static_cast<int64_t>(traversalDims.size());
       ++oldIdx) {
    if (traversalDims[oldIdx].getSize() == 1) continue;
    auto it = traversalPieces.find(oldIdx);
    if (it == traversalPieces.end()) continue;
    for (const TraversalPiece& tp : it->second) {
      // Mixed-radix digit: (i / divBy) mod modBy (modBy 0 => no modulus). A
      // whole-dim piece (divBy 1, modBy 0) leaves the index untouched.
      std::string expr = traversalExprs[oldIdx];
      if (tp.divBy > 1) expr = floorDivExpr(expr, tp.divBy);
      if (tp.modBy > 0) expr = modExpr(expr, tp.modBy);
      if (failed(emitTerm(tp.coeff, expr))) return failure();
    }
  }
  for (int64_t g = 0; g < static_cast<int64_t>(gapDims.size()); ++g) {
    auto it = gapCoeff.find(g);
    if (it != gapCoeff.end() &&
        failed(emitTerm(it->second, "g" + std::to_string(g))))
      return failure();
  }
  for (int64_t e = 0; e < static_cast<int64_t>(replicationDims.size()); ++e) {
    auto it = replicationCoeff.find(e);
    if (it != replicationCoeff.end()) {
      const auto varName =
          "d" + std::to_string(numActiveTraversalComponents + e);
      if (failed(emitTerm(it->second, varName))) return failure();
    }
  }
  return success();
}

static FailureOr<std::string> emitSplitCtSlotIsl(
    int64_t n, size_t prefix, ArrayRef<LayoutPieceKind> pieces,
    ArrayRef<int64_t> pieceIndex, ArrayRef<int64_t> pieceDivBy,
    ArrayRef<int64_t> pieceModBy, const SmallVector<DimAttr>& traversalDims,
    const SmallVector<DimAttr>& replicationDims,
    const SmallVector<DimAttr>& gapDims, int64_t numTraversalComponents,
    int64_t numReplication, int64_t numGap, ArrayRef<int64_t> rolls,
    ArrayRef<int64_t> rollScales, ArrayAttr rotomDims) {
  if (prefix > pieces.size()) return failure();

  const llvm::DenseSet<int64_t> rolledGapIndices =
      rolledGapIndexSet(rolls, rotomDims);

  int64_t numCt = 1;
  for (size_t p = 0; p < prefix; ++p) {
    if (pieces[p] == LayoutPieceKind::Traversal) {
      int64_t sz = traversalDims[pieceIndex[p]].getSize();
      // Effective extent of the piece (the high/ct part sz/L for a straddle).
      sz = (pieceModBy[p] > 0) ? pieceModBy[p] : sz / pieceDivBy[p];
      numCt *= sz;
    } else if (pieces[p] == LayoutPieceKind::Replication) {
      numCt *= replicationDims[pieceIndex[p]].getSize();
    } else if (pieces[p] == LayoutPieceKind::Gap &&
               rolledGapIndices.contains(pieceIndex[p])) {
      // A rolled-by gap claims its blocks (one rotation per block index).
      numCt *= gapDims[pieceIndex[p]].getSize();
    }
  }
  if (numCt < 1) numCt = 1;

  std::string s;
  llvm::raw_string_ostream os(s);

  os << "{ [";
  for (int64_t i = 0; i < numTraversalComponents; ++i) {
    if (i) os << ", ";
    os << "i" << i;
  }
  os << "] -> [ct, slot] : ";

  bool first = true;
  auto emitAnd = [&]() {
    if (!first) os << " and ";
    first = false;
  };

  for (int64_t i = 0; i < numTraversalComponents; ++i) {
    const DimAttr d = traversalDims[i];
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
      if (pieces[p] == LayoutPieceKind::Replication) {
        if (!firstVar) os << ", ";
        firstVar = false;
        os << "d" << (numTraversalComponents + pieceIndex[p]);
      } else if (pieces[p] == LayoutPieceKind::Gap &&
                 rolledGapIndices.contains(pieceIndex[p])) {
        if (!firstVar) os << ", ";
        firstVar = false;
        os << "g" << pieceIndex[p];
      }
    }
    os << " : ";
    first = true;
  }

  emitAnd();
  bool firstTerm = true;
  os << "ct = ";
  if (failed(emitSegmentAddress(
          os, firstTerm, pieces, pieceIndex, pieceDivBy, pieceModBy,
          traversalDims, gapDims, replicationDims, numTraversalComponents, 0,
          prefix, rolledGapIndices, rolls, rollScales, rotomDims)))
    return failure();
  if (firstTerm) os << "0";

  emitAnd();
  firstTerm = true;
  os << "slot = ";
  if (failed(emitSegmentAddress(os, firstTerm, pieces, pieceIndex, pieceDivBy,
                                pieceModBy, traversalDims, gapDims,
                                replicationDims, numTraversalComponents, prefix,
                                pieces.size(), rolledGapIndices, rolls,
                                rollScales, rotomDims)))
    return failure();
  if (firstTerm) os << "0";

  if (numLocalVars > 0) {
    for (int64_t k = 0; k < numReplication; ++k) {
      const DimAttr d = replicationDims[k];
      emitAnd();
      os << "0 <= d" << (numTraversalComponents + k)
         << " <= " << (d.getSize() - 1);
    }
    for (int64_t g = 0; g < numGap; ++g) {
      if (!rolledGapIndices.contains(g)) continue;
      emitAnd();
      os << "0 <= g" << g << " <= " << (gapDims[g].getSize() - 1);
    }
  }

  os << " }";
  os.flush();
  return s;
}

static FailureOr<std::string> lowerToIslImpl(LayoutAttr layout) {
  auto maybeData = preprocessLayoutAttr(layout);
  if (failed(maybeData)) return failure();
  const LayoutData& data = *maybeData;

  llvm::DenseMap<int64_t, llvm::DenseSet<int64_t>> seenDimStride;
  for (DimAttr d : data.traversalDims) {
    auto& seenStridesForDim = seenDimStride[d.getDim()];
    if (seenStridesForDim.contains(d.getStride())) return failure();
    seenStridesForDim.insert(d.getStride());
  }

  const int64_t numTraversalComponents =
      static_cast<int64_t>(data.traversalDims.size());
  const int64_t numReplication =
      static_cast<int64_t>(data.replicationDims.size());
  const int64_t numGap = static_cast<int64_t>(data.gapDims.size());
  DenseI64ArrayAttr rollsAttr = layout.getRolls();
  ArrayRef<int64_t> rolls =
      rollsAttr ? rollsAttr.asArrayRef() : ArrayRef<int64_t>{};
  DenseI64ArrayAttr scalesAttr = layout.getRollScales();
  ArrayRef<int64_t> rollScales =
      scalesAttr ? scalesAttr.asArrayRef() : ArrayRef<int64_t>{};
  return emitSplitCtSlotIsl(
      data.n, data.ctPrefixLen, data.pieces, data.pieceIndex, data.pieceDivBy,
      data.pieceModBy, data.traversalDims, data.replicationDims, data.gapDims,
      numTraversalComponents, numReplication, numGap, rolls, rollScales,
      layout.getDims());
}

}  // namespace

FailureOr<std::string> RotomTensorExtLayoutLowering::lowerToTensorExtIsl(
    LayoutAttr layout) {
  return lowerToIslImpl(layout);
}

}  // namespace mlir::heir::rotom
