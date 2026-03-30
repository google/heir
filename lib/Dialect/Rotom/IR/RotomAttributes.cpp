#include "lib/Dialect/Rotom/IR/RotomAttributes.h"

#include <cstdint>
#include <vector>

#include "llvm/include/llvm/ADT/DenseSet.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"       // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

namespace {

static size_t inferCtPrefixLen(ArrayRef<DimAttr> dims, int64_t n) {
  int64_t nRem = n;
  size_t i = dims.size();
  while (i > 0) {
    if (nRem <= 1) break;
    DimAttr d = dims[i - 1];
    const int64_t sz = d.getSize();
    if (sz <= 0) break;
    if (sz <= nRem && (nRem % sz == 0)) {
      nRem /= sz;
      --i;
      continue;
    }
    break;
  }
  while (i > 0 && dims[i - 1].getSize() == 1) --i;
  return i;
}

static int64_t computeImplicitFrontGap(ArrayRef<DimAttr> dims, int64_t n) {
  int64_t nRem = n;
  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    DimAttr d = *it;
    if (d.isGap()) return 1;
    if (nRem <= 1) break;
    const int64_t sz = d.getSize();
    if (sz <= 0) return 1;
    if (sz <= nRem && nRem % sz == 0) nRem /= sz;
  }
  return nRem;
}

static FailureOr<LayoutData> preprocessLayoutData(ArrayAttr dims, int64_t n,
                                                  MLIRContext *ctx) {
  LayoutData data;
  data.n = n;
  if (data.n <= 0) return failure();

  data.originalDims.reserve(dims.size());
  for (Attribute a : dims) {
    auto d = dyn_cast<DimAttr>(a);
    if (!d) return failure();
    data.originalDims.push_back(d);
    if (d.isGap()) {
      data.pieceIndex.push_back(static_cast<int64_t>(data.gapDims.size()));
      data.gapDims.push_back(d);
      data.pieces.push_back(LayoutPieceKind::Gap);
      continue;
    }
    if (d.isReplicate()) {
      data.pieceIndex.push_back(
          static_cast<int64_t>(data.replicationDims.size()));
      data.replicationDims.push_back(d);
      data.pieces.push_back(LayoutPieceKind::Replication);
      continue;
    }
    if (d.getDim() >= 0) {
      data.pieceIndex.push_back(
          static_cast<int64_t>(data.traversalDims.size()));
      data.traversalDims.push_back(d);
      data.pieces.push_back(LayoutPieceKind::Traversal);
      continue;
    }
    return failure();
  }

  data.ctPrefixLen =
      static_cast<int64_t>(inferCtPrefixLen(data.originalDims, data.n));

  const int64_t implicitFrontGapSize =
      computeImplicitFrontGap(data.originalDims, data.n);
  if (implicitFrontGapSize > 1) {
    const int64_t gapIdx = static_cast<int64_t>(data.gapDims.size());
    data.gapDims.push_back(DimAttr::get(ctx, /*dim=*/-2,
                                        /*size=*/implicitFrontGapSize,
                                        /*stride=*/1));
    data.pieces.insert(data.pieces.begin() + data.ctPrefixLen,
                       LayoutPieceKind::Gap);
    data.pieceIndex.insert(data.pieceIndex.begin() + data.ctPrefixLen, gapIdx);
  }

  llvm::DenseSet<int64_t> seenDim;
  bool allUnique = true;
  for (const DimAttr &d : data.traversalDims) {
    if (seenDim.contains(d.getDim())) {
      allUnique = false;
      break;
    }
    seenDim.insert(d.getDim());
  }
  if (allUnique && data.traversalDims.size() > 1) {
    llvm::SmallVector<std::pair<int64_t, int64_t>> byDim;
    byDim.reserve(data.traversalDims.size());
    for (int64_t i = 0; i < static_cast<int64_t>(data.traversalDims.size());
         ++i) {
      byDim.push_back({data.traversalDims[i].getDim(), i});
    }
    llvm::sort(byDim,
               [](const auto &a, const auto &b) { return a.first < b.first; });

    llvm::SmallVector<DimAttr> reorderedTraversal;
    reorderedTraversal.reserve(data.traversalDims.size());
    llvm::SmallVector<int64_t> oldToNew(data.traversalDims.size(), 0);
    for (int64_t newIdx = 0; newIdx < static_cast<int64_t>(byDim.size());
         ++newIdx) {
      const int64_t oldIdx = byDim[newIdx].second;
      oldToNew[oldIdx] = newIdx;
      reorderedTraversal.push_back(data.traversalDims[oldIdx]);
    }
    data.traversalDims = std::move(reorderedTraversal);

    for (size_t p = 0; p < data.pieces.size(); ++p) {
      if (data.pieces[p] == LayoutPieceKind::Traversal) {
        data.pieceIndex[p] = oldToNew[data.pieceIndex[p]];
      }
    }
  }

  return data;
}

}  // namespace

LogicalResult DimAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              int64_t dim, int64_t size, int64_t stride) {
  if (dim < -2) {
    return emitError() << "`dim` must be >= -2, got " << dim;
  }
  if (size <= 0) {
    return emitError() << "`size` must be > 0, got " << size;
  }
  if (stride <= 0) {
    return emitError() << "`stride` must be > 0, got " << stride;
  }
  return success();
}

FailureOr<LayoutData> preprocessLayoutAttr(LayoutAttr layout) {
  return preprocessLayoutData(layout.getDims(), layout.getN(),
                              layout.getContext());
}

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayAttr dims, int64_t n) {
  if (n <= 0) {
    return emitError() << "`n` must be > 0, got " << n;
  }
  auto preprocessed = preprocessLayoutData(dims, n, dims.getContext());
  if (failed(preprocessed)) {
    return emitError() << "`dims` must be an array of `#rotom.dim<...>`";
  }

  MLIRContext *ctx = dims.getContext();
  std::vector<DimAttr> ctDims;
  std::vector<DimAttr> slotDims;

  int64_t nRem = n;
  for (auto it = preprocessed->originalDims.rbegin();
       it != preprocessed->originalDims.rend(); ++it) {
    DimAttr d = *it;
    const int64_t size = d.getSize();

    if (nRem <= 1) {
      ctDims.insert(ctDims.begin(), d);
      continue;
    }

    // Size > nRem: split into ct and slot dims
    if (size > nRem) {
      if (size % nRem != 0) {
        return emitError() << "dim size " << size
                           << " must be divisible by remaining slot capacity "
                           << nRem;
      }

      slotDims.insert(slotDims.begin(),
                      DimAttr::get(ctx, d.getDim(), nRem, /*stride=*/1));
      ctDims.insert(ctDims.begin(), DimAttr::get(ctx, d.getDim(), size / nRem,
                                                 /*stride=*/nRem));
      nRem /= size;
      continue;
    }

    // Size == nRem: add to slot dims
    if (size == nRem) {
      slotDims.insert(slotDims.begin(), d);
      nRem /= size;
      continue;
    }

    // Size divides nRem: add to slot dims
    if (nRem % size == 0) {
      slotDims.insert(slotDims.begin(), d);
      nRem /= size;
      continue;
    }

    // Size does not divide nRem (e.g. odd input channels): keep in ctDims
    // so slotDims can remain pow2 and nRem is unchanged.
    ctDims.insert(ctDims.begin(), d);
  }

  // If there is remaining slot capacity, insert a gap dim at the front.
  if (nRem > 1) {
    slotDims.insert(slotDims.begin(),
                    DimAttr::get(ctx, /*dim=*/-2, /*size=*/nRem,
                                 /*stride=*/1));
  }

  // Remove gap dims from ctDims.
  std::vector<DimAttr> ctDimsFiltered;
  ctDimsFiltered.reserve(ctDims.size());
  for (DimAttr d : ctDims) {
    if (!d.isGap()) ctDimsFiltered.push_back(d);
  }
  ctDims.swap(ctDimsFiltered);

  // Enforce the Rotom invariant: slot-dim sizes/strides must be powers of 2.
  for (DimAttr d : slotDims) {
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(d.getSize()))) {
      return emitError() << "slot dim size must be a power of two, got "
                         << d.getSize();
    }
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(d.getStride()))) {
      return emitError() << "slot dim stride must be a power of two, got "
                         << d.getStride();
    }
  }

  return success();
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
