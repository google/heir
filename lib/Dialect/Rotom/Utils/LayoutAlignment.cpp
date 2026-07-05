#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Transforms/LayoutOptimization/LayoutConversionCost.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"        // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {
// Collects a layout's dims as DimAttrs for the shared ct/slot-split helper.
static SmallVector<DimAttr> collectDims(LayoutAttr layout) {
  SmallVector<DimAttr> dims;
  dims.reserve(layout.getDims().size());
  for (Attribute attr : layout.getDims()) dims.push_back(cast<DimAttr>(attr));
  return dims;
}

int64_t layoutNumCiphertexts(LayoutAttr layout) {
  SmallVector<DimAttr> dims = collectDims(layout);
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

// Decomposes a layout's slot region into power-of-two "atoms" -- one per bit of
// each traversal axis -- in most-significant-first order, each tagged
// (dim, bit) where bit is the log2 of the bit's within-axis stride. Returns
// nullopt when the slot region holds anything v1 does not analyze: a
// gap/replication piece, or a non-power-of-two extent or stride. Because the
// tag is (dim, bit), a piece and its finer split decompose identically -- e.g.
// [0:4:1] and [0:2:2][0:2:1] both yield (0,1)(0,0).
static std::optional<SmallVector<std::pair<int64_t, int64_t>>>
atomizeSlotRegion(LayoutAttr layout) {
  SmallVector<std::pair<int64_t, int64_t>> atoms;
  size_t ctPrefixLen = inferCtPrefixLen(collectDims(layout), layout.getN());
  ArrayAttr dims = layout.getDims();
  for (size_t i = ctPrefixLen; i < dims.size(); ++i) {
    auto dim = cast<DimAttr>(dims[i]);
    if (dim.isGap() || dim.isReplicate()) return std::nullopt;
    int64_t size = dim.getSize();
    int64_t stride = dim.getStride();
    if (size < 1 || (size & (size - 1)) != 0) return std::nullopt;
    if (stride < 1 || (stride & (stride - 1)) != 0) return std::nullopt;
    int64_t base = static_cast<int64_t>(llvm::Log2_64(stride));
    int64_t numBits = static_cast<int64_t>(llvm::Log2_64(size));
    for (int64_t b = numBits - 1; b >= 0; --b) {
      atoms.push_back({dim.getDim(), base + b});
    }
  }
  return atoms;
}

SmallVector<ConversionMove> conversionMoves(LayoutAttr lhs, LayoutAttr rhs) {
  ConversionMove sentinel{/*dim=*/-1, /*bit=*/-1, /*fromSlot=*/-1,
                          /*toSlot=*/-1};
  if (lhs.getN() != rhs.getN()) return {sentinel};

  std::optional<SmallVector<std::pair<int64_t, int64_t>>> lhsAtoms =
      atomizeSlotRegion(lhs);
  std::optional<SmallVector<std::pair<int64_t, int64_t>>> rhsAtoms =
      atomizeSlotRegion(rhs);
  if (!lhsAtoms || !rhsAtoms || lhsAtoms->size() != rhsAtoms->size()) {
    return {sentinel};
  }

  // Slot-bit position counted from the right (ascending slot weight): the
  // leftmost (most-significant) atom occupies the highest slot position. Both
  // sides pack the same tensor at the same n, so an aligned slot placement
  // means every (dim, bit) sits at the same position on both sides.
  size_t numAtoms = rhsAtoms->size();
  llvm::DenseMap<std::pair<int64_t, int64_t>, int64_t> rhsPos;
  for (size_t i = 0; i < numAtoms; ++i) {
    rhsPos[(*rhsAtoms)[i]] = static_cast<int64_t>(numAtoms - 1 - i);
  }

  SmallVector<ConversionMove> moves;
  for (size_t i = 0; i < numAtoms; ++i) {
    std::pair<int64_t, int64_t> atom = (*lhsAtoms)[i];
    int64_t toSlot = static_cast<int64_t>(numAtoms - 1 - i);
    auto it = rhsPos.find(atom);
    if (it == rhsPos.end()) return {sentinel};  // different slot bit sets
    if (it->second != toSlot) {
      moves.push_back({atom.first, atom.second, it->second, toSlot});
    }
  }
  return moves;
}

std::optional<int64_t> shiftNetworkConversionCost(LayoutAttr from,
                                                  LayoutAttr to) {
  if (from == to) return 0;

  // Bridge each rotom layout to a tensor_ext layout (the same lowering the
  // materializer uses), then reuse the shift-network cost model.
  MLIRContext* ctx = from.getContext();
  FailureOr<std::string> fromIsl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(from);
  FailureOr<std::string> toIsl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(to);
  if (failed(fromIsl) || failed(toIsl)) return std::nullopt;

  auto fromLayout = tensor_ext::LayoutAttr::get(ctx, *fromIsl);
  auto toLayout = tensor_ext::LayoutAttr::get(ctx, *toIsl);
  return computeCostOfLayoutConversion(/*ciphertextSize=*/from.getN(),
                                       fromLayout, toLayout,
                                       /*vveRandomSeed=*/0,
                                       /*vveRandomTries=*/16);
}

FailureOr<SmallVector<LayoutExpansionStep>> planLayoutExpansion(LayoutAttr from,
                                                                LayoutAttr to) {
  if (!from || !to || from.getN() != to.getN()) return failure();
  MLIRContext* ctx = from.getContext();
  FailureOr<std::string> fromIsl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(from);
  FailureOr<std::string> toIsl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(to);
  if (failed(fromIsl) || failed(toIsl)) return failure();
  return planLayoutExpansion(
      tensor_ext::LayoutAttr::get(ctx, *fromIsl).getIntegerRelation(),
      tensor_ext::LayoutAttr::get(ctx, *toIsl).getIntegerRelation(),
      from.getN());
}

FailureOr<SmallVector<LayoutExpansionStep>> planLayoutExpansion(
    const presburger::IntegerRelation& fromRelation,
    const presburger::IntegerRelation& toRelation, int64_t n) {
  if (fromRelation.getNumDomainVars() != toRelation.getNumDomainVars()) {
    return failure();
  }

  // One source placement per tensor point (any replica of a replicated
  // source works; take the first).
  PointPairCollector fromPoints(fromRelation.getNumDomainVars(),
                                /*rangeDims=*/2);
  enumeratePoints(fromRelation, fromPoints);
  std::map<std::vector<int64_t>, std::pair<int64_t, int64_t>> sourceFor;
  for (const auto& [domain, range] : fromPoints.points) {
    sourceFor.try_emplace(domain, std::make_pair(range[0], range[1]));
  }

  // Group every target placement by (targetCt, sourceCt, left-rotation): a
  // left rotation by r maps source slot s onto target slot t = s - r (mod n).
  PointPairCollector toPoints(toRelation.getNumDomainVars(), /*rangeDims=*/2);
  enumeratePoints(toRelation, toPoints);
  std::map<std::tuple<int64_t, int64_t, int64_t>, SmallVector<int64_t>> groups;
  for (const auto& [domain, range] : toPoints.points) {
    auto it = sourceFor.find(domain);
    if (it == sourceFor.end()) return failure();
    const auto [sourceCt, sourceSlot] = it->second;
    const int64_t targetCt = range[0];
    const int64_t targetSlot = range[1];
    const int64_t shift = ((sourceSlot - targetSlot) % n + n) % n;
    groups[{targetCt, sourceCt, shift}].push_back(targetSlot);
  }

  SmallVector<LayoutExpansionStep> steps;
  steps.reserve(groups.size());
  for (auto& [key, targetSlots] : groups) {
    auto [targetCt, sourceCt, shift] = key;
    llvm::sort(targetSlots);
    steps.push_back(
        LayoutExpansionStep{targetCt, sourceCt, shift, std::move(targetSlots)});
  }
  return steps;
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
  // Differing ciphertext counts are allowed: a same-count alignment lowers to
  // tensor_ext.convert_layout, a count-changing one to the explicit
  // rotate/mask/accumulate steps of planLayoutExpansion.
  return isMaterializableRotomLayout(lhsLayout) &&
         isMaterializableRotomLayout(rhsLayout) &&
         isMaterializableRotomLayout(resultLayout);
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
