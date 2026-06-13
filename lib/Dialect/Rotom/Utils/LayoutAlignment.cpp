#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {
namespace {

struct DimComponent {
  int64_t size;
  int64_t stride;
  bool ciphertextSide;
};

SmallVector<DimComponent> getDimComponents(LayoutAttr layout,
                                           int64_t logicalDim) {
  SmallVector<DimComponent> components;
  size_t ctPrefixLen = inferCtPrefixLen(layout);
  for (auto [index, attr] : llvm::enumerate(layout.getDims())) {
    auto dim = cast<DimAttr>(attr);
    if (dim.isGap() || dim.isReplicate()) continue;
    if (dim.getDim() != logicalDim) continue;
    components.push_back({dim.getSize(), dim.getStride(),
                          static_cast<size_t>(index) < ctPrefixLen});
  }
  return components;
}

bool componentsEqual(ArrayRef<DimComponent> lhs, ArrayRef<DimComponent> rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto [lhsComponent, rhsComponent] : llvm::zip(lhs, rhs)) {
    if (lhsComponent.size != rhsComponent.size) return false;
    if (lhsComponent.stride != rhsComponent.stride) return false;
    if (lhsComponent.ciphertextSide != rhsComponent.ciphertextSide) {
      return false;
    }
  }
  return true;
}

}  // namespace

size_t inferCtPrefixLen(LayoutAttr layout) {
  ArrayAttr dims = layout.getDims();
  int64_t nRem = layout.getN();
  size_t prefix = dims.size();
  while (prefix > 0) {
    if (nRem <= 1) break;
    auto dim = cast<DimAttr>(dims[prefix - 1]);
    int64_t size = dim.getSize();
    if (size <= 0) break;
    if (size <= nRem && nRem % size == 0) {
      nRem /= size;
      --prefix;
      continue;
    }
    break;
  }
  while (prefix > 0 && cast<DimAttr>(dims[prefix - 1]).getSize() == 1) {
    --prefix;
  }
  return prefix;
}

int64_t layoutNumCiphertexts(LayoutAttr layout) {
  int64_t numCt = 1;
  size_t ctPrefixLen = inferCtPrefixLen(layout);
  ArrayAttr dims = layout.getDims();
  for (size_t i = 0; i < ctPrefixLen; ++i) {
    auto dim = cast<DimAttr>(dims[i]);
    if (dim.isGap()) continue;
    numCt *= dim.getSize();
  }
  return std::max<int64_t>(numCt, 1);
}

bool dimensionsAligned(LayoutAttr lhsLayout, int64_t lhsDim,
                       LayoutAttr rhsLayout, int64_t rhsDim) {
  return componentsEqual(getDimComponents(lhsLayout, lhsDim),
                         getDimComponents(rhsLayout, rhsDim));
}

bool layoutsAlignedByDimMap(LayoutAttr lhsLayout, LayoutAttr rhsLayout,
                            ArrayRef<std::pair<int64_t, int64_t>> dimMap) {
  if (lhsLayout.getN() != rhsLayout.getN()) return false;
  for (auto [lhsDim, rhsDim] : dimMap) {
    if (!dimensionsAligned(lhsLayout, lhsDim, rhsLayout, rhsDim)) {
      return false;
    }
  }
  return true;
}

bool hasOnlyUnitStridedTraversalDims(LayoutAttr layout) {
  for (Attribute attr : layout.getDims()) {
    auto dim = cast<DimAttr>(attr);
    // Replication is not a traversal/rolling dimension. Its stride describes
    // replica placement, so it is allowed here even when the stride is not one.
    if (dim.isGap() || dim.isReplicate()) continue;
    if (dim.getStride() != 1) return false;
  }
  return true;
}

bool isMaterializableRotomLayout(LayoutAttr layout) {
  return succeeded(RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout));
}

bool supportsRotomAlignmentLowering(LayoutAttr lhsLayout, LayoutAttr rhsLayout,
                                    LayoutAttr resultLayout) {
  if (lhsLayout.getN() != rhsLayout.getN() ||
      lhsLayout.getN() != resultLayout.getN()) {
    return false;
  }
  if (!hasOnlyUnitStridedTraversalDims(lhsLayout) ||
      !hasOnlyUnitStridedTraversalDims(rhsLayout) ||
      !hasOnlyUnitStridedTraversalDims(resultLayout)) {
    return false;
  }
  if (!isMaterializableRotomLayout(lhsLayout) ||
      !isMaterializableRotomLayout(rhsLayout) ||
      !isMaterializableRotomLayout(resultLayout)) {
    return false;
  }

  // The current Rotom lowering aligns logical scalar contributions by masking
  // and remapping between ciphertext-semantic tensors. tensor_ext.remap remaps
  // within one ciphertext tensor shape, so all participating layouts must
  // materialize to the same number of ciphertexts.
  return layoutNumCiphertexts(lhsLayout) == layoutNumCiphertexts(rhsLayout) &&
         layoutNumCiphertexts(lhsLayout) == layoutNumCiphertexts(resultLayout);
}

LayoutAttr withRolls(LayoutAttr layout,
                     ArrayRef<std::pair<int64_t, int64_t>> rolls) {
  MLIRContext* ctx = layout.getContext();
  SmallVector<int64_t> flat;
  flat.reserve(rolls.size() * 2);
  for (auto [from, to] : rolls) {
    flat.push_back(from);
    flat.push_back(to);
  }
  return LayoutAttr::get(ctx, layout.getDims(), layout.getN(),
                         DenseI64ArrayAttr::get(ctx, flat));
}

SmallVector<LayoutAttr> enumerateSingleRolls(LayoutAttr layout) {
  SmallVector<LayoutAttr> variants;
  ArrayAttr dims = layout.getDims();
  int64_t numDims = static_cast<int64_t>(dims.size());

  // Rolls already on the layout are preserved; the new roll is appended after
  // them so existing roll metadata keeps its left-to-right application order.
  SmallVector<std::pair<int64_t, int64_t>> baseRolls;
  if (DenseI64ArrayAttr existing = layout.getRolls()) {
    ArrayRef<int64_t> flat = existing.asArrayRef();
    for (size_t i = 0; i + 1 < flat.size(); i += 2) {
      baseRolls.push_back({flat[i], flat[i + 1]});
    }
  }

  // Rolls are only applied to the slot line of the materialized layout, so a
  // roll whose `from` dim is ciphertext-side modifies an address term that is
  // never emitted -- it lowers identically to the base and carries misleading
  // metadata. Restrict both roll indices to the slot side so every variant is a
  // genuinely distinct packing. (Cross-ciphertext rolls are a future
  // extension.)
  int64_t ctPrefixLen = static_cast<int64_t>(inferCtPrefixLen(layout));
  auto isSlotTraversal = [&](int64_t index, DimAttr dim) {
    return index >= ctPrefixLen && dim && !dim.isGap() && !dim.isReplicate();
  };

  for (int64_t from = ctPrefixLen; from < numDims; ++from) {
    auto fromDim = dyn_cast<DimAttr>(dims[from]);
    if (!isSlotTraversal(from, fromDim)) continue;
    for (int64_t to = ctPrefixLen; to < numDims; ++to) {
      if (from == to) continue;
      auto toDim = dyn_cast<DimAttr>(dims[to]);
      if (!isSlotTraversal(to, toDim)) continue;
      if (fromDim.getSize() != toDim.getSize()) continue;

      SmallVector<std::pair<int64_t, int64_t>> rolls(baseRolls);
      rolls.push_back({from, to});
      LayoutAttr candidate = withRolls(layout, rolls);
      if (isMaterializableRotomLayout(candidate)) {
        variants.push_back(candidate);
      }
    }
  }
  return variants;
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
