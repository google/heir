#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::rotom {
namespace {

static LogicalResult emitSegmentAddress(
    llvm::raw_ostream& os, bool& firstTerm, ArrayRef<LayoutPieceKind> pieces,
    ArrayRef<int64_t> pieceIndex, const SmallVector<DimAttr>& traversalDims,
    const SmallVector<DimAttr>& gapDims,
    const SmallVector<DimAttr>& replicationDims,
    int64_t numActiveTraversalComponents, size_t segStart, size_t segEnd,
    bool foldGapVarsToZero) {
  llvm::SmallVector<int64_t> suffixCoeff(pieces.size(), 0);
  int64_t suffix = 1;
  for (size_t p = segEnd; p > segStart;) {
    --p;
    suffixCoeff[p] = suffix;
    const DimAttr d =
        pieces[p] == LayoutPieceKind::Traversal ? traversalDims[pieceIndex[p]]
        : pieces[p] == LayoutPieceKind::Gap ? gapDims[pieceIndex[p]]
                                            : replicationDims[pieceIndex[p]];
    suffix *= d.getSize();
  }

  auto emitTerm = [&](int64_t coeff, llvm::StringRef var) -> LogicalResult {
    if (coeff == 0) return failure();
    if (!firstTerm) os << " + ";
    firstTerm = false;
    if (coeff == 1) {
      os << var;
    } else {
      os << coeff << " * " << var;
    }
    return success();
  };

  llvm::DenseMap<int64_t, int64_t> traversalCoeff;
  llvm::DenseMap<int64_t, int64_t> gapCoeff;
  llvm::DenseMap<int64_t, int64_t> replicationCoeff;
  for (size_t p = segStart; p < segEnd; ++p) {
    const int64_t coeff = suffixCoeff[p];

    if (pieces[p] == LayoutPieceKind::Traversal) {
      const int64_t ti = pieceIndex[p];
      if (traversalDims[ti].getSize() == 1) continue;
      traversalCoeff[ti] = coeff;
    } else if (pieces[p] == LayoutPieceKind::Gap) {
      if (foldGapVarsToZero) continue;
      const int64_t gk = pieceIndex[p];
      gapCoeff[gk] = coeff;
    } else {
      const int64_t ek = pieceIndex[p];
      replicationCoeff[ek] = coeff;
    }
  }

  for (int64_t oldIdx = 0; oldIdx < static_cast<int64_t>(traversalDims.size());
       ++oldIdx) {
    if (traversalDims[oldIdx].getSize() == 1) continue;
    auto it = traversalCoeff.find(oldIdx);
    if (it != traversalCoeff.end()) {
      if (failed(emitTerm(it->second, "i" + std::to_string(oldIdx))))
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
    int64_t numReplication, int64_t numGap) {
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
                                foldGapVarsToZero)))
    return failure();
  if (firstTerm) os << "0";

  emitAnd();
  firstTerm = true;
  os << "slot = ";
  if (failed(emitSegmentAddress(os, firstTerm, pieces, pieceIndex,
                                traversalDims, gapDims, replicationDims,
                                numTraversalComponents, prefix, pieces.size(),
                                foldGapVarsToZero)))
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
  return emitSplitCtSlotIsl(data.n, data.ctPrefixLen, data.pieces,
                            data.pieceIndex, data.traversalDims,
                            data.replicationDims, data.gapDims,
                            numTraversalComponents, numReplication, numGap);
}

}  // namespace

FailureOr<std::string> RotomTensorExtLayoutLowering::lowerToTensorExtIsl(
    LayoutAttr layout) {
  return lowerToIslImpl(layout);
}

}  // namespace mlir::heir::rotom
