#include "lib/Dialect/Rotom/Utils/RotomLayout.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

SmallVector<DimAttr> layoutDims(LayoutAttr layout) {
  SmallVector<DimAttr> dims;
  dims.reserve(layout.getDims().size());
  for (Attribute attr : layout.getDims()) dims.push_back(cast<DimAttr>(attr));
  return dims;
}

int64_t layoutNumCiphertexts(LayoutAttr layout) {
  SmallVector<DimAttr> dims = layoutDims(layout);
  size_t ctPrefixLen = inferCtPrefixLen(dims, layout.getN());
  // A gap that is a roll-by target claims its blocks (one rotation of the
  // rolled dim per block index), so it counts toward distinct ciphertexts;
  // plain gaps are unclaimed space.
  llvm::DenseSet<int64_t> rolledByPositions;
  if (DenseI64ArrayAttr rolls = layout.getRolls()) {
    ArrayRef<int64_t> r = rolls.asArrayRef();
    for (size_t i = 0; i + 1 < r.size(); i += 2) {
      rolledByPositions.insert(r[i + 1]);
    }
  }
  int64_t numCt = 1;
  for (size_t i = 0; i < ctPrefixLen; ++i) {
    if (dims[i].isGap() &&
        !rolledByPositions.contains(static_cast<int64_t>(i))) {
      continue;
    }
    numCt *= std::max<int64_t>(dims[i].getSize(), 1);
  }
  return std::max<int64_t>(numCt, 1);
}

LayoutAttr mergeAdjacentLayoutDims(LayoutAttr layout) {
  if (!layout) return layout;
  MLIRContext* ctx = layout.getContext();
  SmallVector<DimAttr> dims = layoutDims(layout);
  const size_t ctPrefixLen = inferCtPrefixLen(dims, layout.getN());

  SmallVector<RollSpec> rolls = getRollSpecs(layout);
  // A piece argument reads exactly its piece's part of the axis index; joining
  // that piece with a neighbor would change what the roll rewrites (FROM) or
  // the part it shifts by (BY), so argument pieces keep their identity.
  llvm::SmallDenseSet<int64_t> pinned;
  for (const RollSpec& roll : rolls) {
    if (!roll.from.isAxis) pinned.insert(roll.from.index);
    if (!roll.by.isAxis) pinned.insert(roll.by.index);
  }

  SmallVector<DimAttr> merged;
  SmallVector<int64_t> mergedIndexOf(dims.size());
  auto mergeRegion = [&](size_t begin, size_t end) {
    size_t i = begin;
    while (i < end) {
      DimAttr piece = dims[i];
      // Gap and replication pieces enumerate copies/space in traversal order
      // regardless of stride; traversal pieces join only when the outer piece
      // reads the parts directly above the inner one (one contiguous piece).
      const bool special = piece.isGap() || piece.isReplicate();
      int64_t size = piece.getSize();
      int64_t stride = special ? 1 : piece.getStride();
      mergedIndexOf[i] = merged.size();
      size_t j = i + 1;
      for (; j < end && !pinned.contains(static_cast<int64_t>(i)) &&
             !pinned.contains(static_cast<int64_t>(j));
           ++j) {
        DimAttr next = dims[j];
        if (next.getDim() != piece.getDim()) break;
        if (!special && stride != next.getSize() * next.getStride()) break;
        size *= next.getSize();
        if (!special) stride = next.getStride();
        mergedIndexOf[j] = mergedIndexOf[i];
      }
      merged.push_back(DimAttr::get(ctx, piece.getDim(), size, stride));
      i = j;
    }
  };
  mergeRegion(0, ctPrefixLen);
  mergeRegion(ctPrefixLen, dims.size());
  if (merged.size() == dims.size()) return layout;

  // Re-index piece arguments (pinned, so each survives as its own merged
  // piece) and restate an axis argument as the piece when its axis is no
  // longer split (the canonical form of an unsplit axis).
  auto remapRollArg = [&](RollArg e) -> int64_t {
    if (!e.isAxis) return mergedIndexOf[e.index];
    SmallVector<int64_t> positions;
    for (auto [pos, piece] : llvm::enumerate(merged)) {
      if (!piece.isGap() && !piece.isReplicate() && piece.getDim() == e.index) {
        positions.push_back(pos);
      }
    }
    if (positions.size() == 1) return positions.front();
    return encodeRollArg(e);
  };
  SmallVector<int64_t> rollStorage;
  for (const RollSpec& roll : rolls) {
    rollStorage.push_back(remapRollArg(roll.from));
    rollStorage.push_back(remapRollArg(roll.by));
  }

  canonicalizeLayoutDims(ctx, merged, layout.getN(), rollStorage);
  SmallVector<Attribute> attrs(merged.begin(), merged.end());
  auto dimsAttr = ArrayAttr::get(ctx, attrs);
  auto rollsAttr = DenseI64ArrayAttr::get(ctx, rollStorage);
  // Defensive: a form whose merge does not verify keys as itself.
  ScopedDiagnosticHandler silence(ctx, [](Diagnostic&) { return success(); });
  auto swallow = mlir::detail::getDefaultDiagnosticEmitFn(UnknownLoc::get(ctx));
  if (failed(LayoutAttr::verify(swallow, dimsAttr, layout.getN(), rollsAttr))) {
    return layout;
  }
  return LayoutAttr::get(ctx, dimsAttr, layout.getN(), rollsAttr);
}

bool isMaterializableRotomLayout(LayoutAttr layout) {
  return succeeded(RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout));
}

bool isLowerableRotomLayout(LayoutAttr layout) {
  return layout && isMaterializableRotomLayout(layout);
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
