#include "lib/Dialect/Rotom/Utils/ContractionAlignment.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

// Builds the layout only if it passes LayoutAttr verification; enumerated
// variants that violate an invariant (e.g. a non-power-of-two slot extent)
// are skipped, not diagnosed.
static LayoutAttr makeCheckedLayout(MLIRContext* ctx, ArrayRef<DimAttr> dims,
                                    int64_t n) {
  SmallVector<Attribute> attrs(dims.begin(), dims.end());
  auto dimsAttr = ArrayAttr::get(ctx, attrs);
  auto rolls = DenseI64ArrayAttr::get(ctx, ArrayRef<int64_t>{});
  auto swallow = []() { return InFlightDiagnostic(); };
  if (failed(LayoutAttr::verify(swallow, dimsAttr, n, rolls))) return {};
  return LayoutAttr::get(ctx, dimsAttr, n, rolls);
}

// Product of the traversal-piece extents of one iteration dim (1 if absent).
static int64_t iterationDimExtent(LayoutAttr layout, int64_t dim) {
  int64_t extent = 1;
  for (Attribute attr : layout.getDims()) {
    auto piece = cast<DimAttr>(attr);
    if (piece.getDim() == dim) extent *= piece.getSize();
  }
  return extent;
}

// The compute placement with `replacedDim`'s traversal pieces turned into
// replication pieces of the same extent at the same position, so the
// expanded layout keeps the compute layout's footprint piece for piece.
static SmallVector<DimAttr> replaceDimWithReplication(
    MLIRContext* ctx, ArrayRef<DimAttr> computeDims, int64_t replacedDim) {
  SmallVector<DimAttr> result;
  result.reserve(computeDims.size());
  for (DimAttr piece : computeDims) {
    if (piece.getDim() == replacedDim) {
      result.push_back(
          DimAttr::get(ctx, /*dim=*/-1, piece.getSize(), /*stride=*/1));
    } else {
      result.push_back(piece);
    }
  }
  return result;
}

// Rotations/adds to fill the replication pieces that replaced `replacedDim`:
// free in the ciphertext prefix (replicated ciphertexts share a handle),
// log2(extent) rotate-and-adds per distinct ciphertext in the slot region.
static void replicationFillCounts(ArrayRef<DimAttr> computeDims,
                                  ArrayRef<DimAttr> expandedDims,
                                  size_t ctPrefixLen, int64_t numCtCompute,
                                  int64_t replacedDim, int64_t& rotations,
                                  int64_t& adds) {
  int64_t slotFillExtent = 1;
  int64_t ctReplicationExtent = 1;
  for (size_t p = 0; p < computeDims.size(); ++p) {
    if (p < ctPrefixLen) {
      // Any ciphertext-prefix replication (inserted or pre-existing) means
      // the operand has fewer distinct ciphertexts to fill.
      if (expandedDims[p].isReplicate()) {
        ctReplicationExtent *= expandedDims[p].getSize();
      }
      continue;
    }
    if (computeDims[p].getDim() == replacedDim) {
      slotFillExtent *= computeDims[p].getSize();
    }
  }
  int64_t distinctCts = numCtCompute / ctReplicationExtent;
  int64_t steps = static_cast<int64_t>(
      llvm::Log2_64(static_cast<uint64_t>(slotFillExtent)));
  rotations = distinctCts * steps;
  adds = distinctCts * steps;
}

SmallVector<MatmulPlan> enumerateMatmulPlans(LayoutAttr lhs, LayoutAttr rhs) {
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return {};
  MLIRContext* ctx = lhs.getContext();
  const int64_t n = lhs.getN();
  const int64_t iExtent = iterationDimExtent(lhs, kMatmulDimI);
  const int64_t jExtent = iterationDimExtent(rhs, kMatmulDimJ);

  // Candidate compute placements: host one operand unchanged and insert the
  // missing free dim innermost (slot region) and outermost (ct region).
  SmallVector<SmallVector<DimAttr>> computeVariants;
  auto addHostVariants = [&](LayoutAttr host, int64_t freeDim,
                             int64_t freeExtent) {
    SmallVector<DimAttr> base;
    for (Attribute attr : host.getDims()) base.push_back(cast<DimAttr>(attr));
    if (freeExtent == 1) {
      computeVariants.push_back(base);
      return;
    }
    DimAttr freePiece = DimAttr::get(ctx, freeDim, freeExtent, /*stride=*/1);
    SmallVector<DimAttr> slotVariant = base;
    slotVariant.push_back(freePiece);
    computeVariants.push_back(std::move(slotVariant));
    SmallVector<DimAttr> ctVariant = {freePiece};
    ctVariant.append(base.begin(), base.end());
    computeVariants.push_back(std::move(ctVariant));
  };
  addHostVariants(lhs, kMatmulDimJ, jExtent);
  addHostVariants(rhs, kMatmulDimI, iExtent);

  llvm::DenseSet<Attribute> seenComputeLayouts;
  SmallVector<MatmulPlan> plans;
  for (const SmallVector<DimAttr>& computeDims : computeVariants) {
    LayoutAttr computeLayout = makeCheckedLayout(ctx, computeDims, n);
    if (!computeLayout) continue;
    if (!seenComputeLayouts.insert(computeLayout).second) continue;

    const size_t ctPrefixLen = inferCtPrefixLen(computeDims, n);
    const int64_t numCtCompute = layoutNumCiphertexts(computeLayout);

    SmallVector<DimAttr> expandedLhsDims =
        replaceDimWithReplication(ctx, computeDims, kMatmulDimJ);
    SmallVector<DimAttr> expandedRhsDims =
        replaceDimWithReplication(ctx, computeDims, kMatmulDimI);

    // Sum away k: slot pieces become replication (the reduced value is left
    // in every k-slot), ciphertext pieces collapse into adds and are dropped.
    SmallVector<DimAttr> resultDims;
    int64_t kSlotExtent = 1;
    int64_t kCtExtent = 1;
    for (size_t p = 0; p < computeDims.size(); ++p) {
      DimAttr piece = computeDims[p];
      if (piece.getDim() != kMatmulDimK) {
        resultDims.push_back(piece);
        continue;
      }
      if (p < ctPrefixLen) {
        kCtExtent *= piece.getSize();
        continue;
      }
      kSlotExtent *= piece.getSize();
      resultDims.push_back(
          DimAttr::get(ctx, /*dim=*/-1, piece.getSize(), /*stride=*/1));
    }

    MatmulPlan plan;
    plan.computeLayout = computeLayout;
    plan.expandedLhs = makeCheckedLayout(ctx, expandedLhsDims, n);
    plan.expandedRhs = makeCheckedLayout(ctx, expandedRhsDims, n);
    plan.resultLayout = makeCheckedLayout(ctx, resultDims, n);
    if (!plan.expandedLhs || !plan.expandedRhs || !plan.resultLayout) continue;

    replicationFillCounts(computeDims, expandedLhsDims, ctPrefixLen,
                          numCtCompute, kMatmulDimJ, plan.lhsFillRotations,
                          plan.lhsFillAdds);
    replicationFillCounts(computeDims, expandedRhsDims, ctPrefixLen,
                          numCtCompute, kMatmulDimI, plan.rhsFillRotations,
                          plan.rhsFillAdds);

    const int64_t numCtResult = numCtCompute / kCtExtent;
    const int64_t reduceSteps =
        static_cast<int64_t>(llvm::Log2_64(static_cast<uint64_t>(kSlotExtent)));
    plan.reduceRotations = numCtResult * reduceSteps;
    plan.reduceAdds = numCtResult * reduceSteps + (numCtCompute - numCtResult);

    plans.push_back(std::move(plan));
  }
  return plans;
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
