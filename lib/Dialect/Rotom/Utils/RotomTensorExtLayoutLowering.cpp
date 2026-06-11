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

/// Maps a `#rotom.dim` from the layout's `dims` list to its iterator index `i*`
/// after preprocessing (match logical axis, size, and stride).
static FailureOr<int64_t> traversalIndexForRotomDim(
    const SmallVector<DimAttr>& traversalDims, DimAttr want) {
  for (int64_t i = 0; i < static_cast<int64_t>(traversalDims.size()); ++i) {
    if (traversalDims[i].getDim() == want.getDim() &&
        traversalDims[i].getSize() == want.getSize() &&
        traversalDims[i].getStride() == want.getStride()) {
      return i;
    }
  }
  return failure();
}

static std::string modExpr(llvm::StringRef expr, int64_t mod) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "(" << expr << " - " << mod << " * floor((" << expr << ") / " << mod
     << "))";
  return out;
}

static LogicalResult emitSegmentAddress(
    llvm::raw_ostream& os, bool& firstTerm, ArrayRef<LayoutPieceKind> pieces,
    ArrayRef<int64_t> pieceIndex, const SmallVector<DimAttr>& traversalDims,
    const SmallVector<DimAttr>& gapDims,
    const SmallVector<DimAttr>& replicationDims,
    int64_t numActiveTraversalComponents, size_t segStart, size_t segEnd,
    bool foldGapVarsToZero, ArrayRef<int64_t> rolls, ArrayAttr rotomDims,
    bool isSlotLine) {
  llvm::SmallVector<int64_t> suffixCoeff(pieces.size(), 0);
  int64_t suffix = 1;
  for (size_t p = segEnd; p > segStart;) {
    --p;
    suffixCoeff[p] = suffix;
    DimAttr d;
    switch (pieces[p]) {
      case LayoutPieceKind::Traversal:
        d = traversalDims[pieceIndex[p]];
        break;
      case LayoutPieceKind::Gap:
        d = gapDims[pieceIndex[p]];
        break;
      case LayoutPieceKind::Replication:
        d = replicationDims[pieceIndex[p]];
        break;
    }
    suffix *= d.getSize();
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
      if (foldGapVarsToZero) continue;
      gapCoeff[pieceIndex[p]] = coeff;
    } else if (pieces[p] == LayoutPieceKind::Replication) {
      replicationCoeff[pieceIndex[p]] = coeff;
    }
  }

  llvm::DenseMap<int64_t, int64_t> traversalCoeff;
  for (size_t p = segStart; p < segEnd; ++p) {
    if (pieces[p] != LayoutPieceKind::Traversal) continue;
    const int64_t ti = pieceIndex[p];
    if (traversalDims[ti].getSize() == 1) continue;
    traversalCoeff[ti] = suffixCoeff[p];
  }

  llvm::SmallVector<std::string> traversalExprs;
  traversalExprs.reserve(traversalDims.size());
  for (int64_t i = 0; i < static_cast<int64_t>(traversalDims.size()); ++i) {
    traversalExprs.push_back("i" + std::to_string(i));
  }

  // Apply roll(a,b) transforms left-to-right:
  // t_a <- (t_a - t_b) mod extent(a).
  if (isSlotLine && !rolls.empty()) {
    if (!rotomDims || rolls.size() % 2 != 0) return failure();
    for (size_t i = 0; i < rolls.size(); i += 2) {
      const int64_t fromIdx = rolls[i];
      const int64_t toIdx = rolls[i + 1];
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
      FailureOr<int64_t> maybeToTrav =
          traversalIndexForRotomDim(traversalDims, toDim);
      if (failed(maybeFromTrav) || failed(maybeToTrav)) return failure();
      const int64_t fromTrav = *maybeFromTrav;
      const int64_t toTrav = *maybeToTrav;
      std::string diffExpr =
          "(" + traversalExprs[fromTrav] + " - " + traversalExprs[toTrav] + ")";
      traversalExprs[fromTrav] = modExpr(diffExpr, fromDim.getSize());
    }
  }

  for (int64_t oldIdx = 0; oldIdx < static_cast<int64_t>(traversalDims.size());
       ++oldIdx) {
    if (traversalDims[oldIdx].getSize() == 1) continue;
    auto it = traversalCoeff.find(oldIdx);
    if (it != traversalCoeff.end()) {
      if (failed(emitTerm(it->second, traversalExprs[oldIdx])))
        return failure();
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
    ArrayRef<int64_t> pieceIndex, const SmallVector<DimAttr>& traversalDims,
    const SmallVector<DimAttr>& replicationDims,
    const SmallVector<DimAttr>& gapDims, int64_t numTraversalComponents,
    int64_t numReplication, int64_t numGap, ArrayRef<int64_t> rolls,
    ArrayAttr rotomDims) {
  if (prefix > pieces.size()) return failure();

  int64_t numCt = 1;
  for (size_t p = 0; p < prefix; ++p) {
    if (pieces[p] == LayoutPieceKind::Traversal) {
      numCt *= traversalDims[pieceIndex[p]].getSize();
    } else if (pieces[p] == LayoutPieceKind::Replication) {
      numCt *= replicationDims[pieceIndex[p]].getSize();
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

  const bool foldGapVarsToZero = numGap > 0;
  const int64_t numLocalVars = numReplication;

  if (numLocalVars > 0) {
    os << " and exists ";
    bool firstVar = true;
    for (size_t p = 0; p < pieces.size(); ++p) {
      if (pieces[p] == LayoutPieceKind::Replication) {
        if (!firstVar) os << ", ";
        firstVar = false;
        os << "d" << (numTraversalComponents + pieceIndex[p]);
      }
    }
    os << " : ";
    first = true;
  }

  emitAnd();
  bool firstTerm = true;
  os << "ct = ";
  if (failed(emitSegmentAddress(os, firstTerm, pieces, pieceIndex,
                                traversalDims, gapDims, replicationDims,
                                numTraversalComponents, 0, prefix,
                                foldGapVarsToZero, rolls, rotomDims,
                                /*isSlotLine=*/false)))
    return failure();
  if (firstTerm) os << "0";

  emitAnd();
  firstTerm = true;
  os << "slot = ";
  if (failed(emitSegmentAddress(os, firstTerm, pieces, pieceIndex,
                                traversalDims, gapDims, replicationDims,
                                numTraversalComponents, prefix, pieces.size(),
                                foldGapVarsToZero, rolls, rotomDims,
                                /*isSlotLine=*/true)))
    return failure();
  if (firstTerm) os << "0";

  if (numLocalVars > 0) {
    for (int64_t k = 0; k < numReplication; ++k) {
      const DimAttr d = replicationDims[k];
      emitAnd();
      os << "0 <= d" << (numTraversalComponents + k)
         << " <= " << (d.getSize() - 1);
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
  return emitSplitCtSlotIsl(
      data.n, data.ctPrefixLen, data.pieces, data.pieceIndex,
      data.traversalDims, data.replicationDims, data.gapDims,
      numTraversalComponents, numReplication, numGap, rolls, layout.getDims());
}

}  // namespace

FailureOr<std::string> RotomTensorExtLayoutLowering::lowerToTensorExtIsl(
    LayoutAttr layout) {
  return lowerToIslImpl(layout);
}

}  // namespace mlir::heir::rotom
