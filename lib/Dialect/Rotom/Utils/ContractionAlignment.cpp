#include "lib/Dialect/Rotom/Utils/ContractionAlignment.h"

#include <cassert>
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
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
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
                                    int64_t n,
                                    ArrayRef<int64_t> rollPairs = {}) {
  SmallVector<Attribute> attrs(dims.begin(), dims.end());
  auto dimsAttr = ArrayAttr::get(ctx, attrs);
  auto rolls = DenseI64ArrayAttr::get(ctx, rollPairs);
  // A real (but silenced) diagnostic emitter: streaming into an inactive
  // InFlightDiagnostic asserts, and an invalid enumerated variant is skipped,
  // not reported.
  ScopedDiagnosticHandler silence(ctx, [](Diagnostic&) { return success(); });
  auto swallow = mlir::detail::getDefaultDiagnosticEmitFn(UnknownLoc::get(ctx));
  if (failed(LayoutAttr::verify(swallow, dimsAttr, n, rolls))) return {};
  return LayoutAttr::get(ctx, dimsAttr, n, rolls);
}

static SmallVector<DimAttr> collectDims(LayoutAttr layout) {
  SmallVector<DimAttr> dims;
  dims.reserve(layout.getDims().size());
  for (Attribute attr : layout.getDims()) dims.push_back(cast<DimAttr>(attr));
  return dims;
}

// Relabels traversal dims through `oldToNew` (gap/replication pieces pass
// through). Every traversal dim present must map to a non-negative new dim.
static SmallVector<DimAttr> relabelTraversalDims(MLIRContext* ctx,
                                                 ArrayRef<DimAttr> dims,
                                                 ArrayRef<int64_t> oldToNew) {
  SmallVector<DimAttr> result;
  result.reserve(dims.size());
  for (DimAttr piece : dims) {
    if (piece.isGap() || piece.isReplicate()) {
      result.push_back(piece);
      continue;
    }
    int64_t oldDim = piece.getDim();
    assert(oldDim >= 0 && oldDim < static_cast<int64_t>(oldToNew.size()) &&
           "traversal dim out of range for the relabeling");
    int64_t newDim = oldToNew[oldDim];
    assert(newDim >= 0 && "relabeling must not drop traversal dims");
    result.push_back(
        DimAttr::get(ctx, newDim, piece.getSize(), piece.getStride()));
  }
  return result;
}

// Product of the traversal-piece extents of one iteration dim (1 if absent).
static int64_t iterationDimExtent(ArrayRef<DimAttr> dims, int64_t dim) {
  int64_t extent = 1;
  for (DimAttr piece : dims) {
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

  // Relabel the operands' own dims into the iteration space: A is (i, k),
  // B is (k, j).
  const SmallVector<int64_t> lhsToIter = {kMatmulDimI, kMatmulDimK};
  const SmallVector<int64_t> rhsToIter = {kMatmulDimK, kMatmulDimJ};
  SmallVector<DimAttr> lhsIter =
      relabelTraversalDims(ctx, collectDims(lhs), lhsToIter);
  SmallVector<DimAttr> rhsIter =
      relabelTraversalDims(ctx, collectDims(rhs), rhsToIter);
  const int64_t iExtent = iterationDimExtent(lhsIter, kMatmulDimI);
  const int64_t jExtent = iterationDimExtent(rhsIter, kMatmulDimJ);

  // ... and back out to each tensor's own dims.
  const SmallVector<int64_t> iterToLhs = {0, -1, 1};
  const SmallVector<int64_t> iterToRhs = {-1, 1, 0};
  const SmallVector<int64_t> iterToResult = {0, 1, -1};

  // Candidate compute placements: host one operand unchanged and insert the
  // missing free dim innermost (slot region), outermost (ct region), or in
  // place of an existing same-extent replication piece (the reverse of the
  // subsumption that derives the expanded operands -- so an operand already
  // at an expanded placement enumerates the compute placement it came from).
  SmallVector<SmallVector<DimAttr>> computeFootprints;
  auto addHostVariants = [&](ArrayRef<DimAttr> host, int64_t freeDim,
                             int64_t freeExtent) {
    SmallVector<DimAttr> base(host.begin(), host.end());
    if (freeExtent == 1) {
      computeFootprints.push_back(base);
      return;
    }
    DimAttr freePiece = DimAttr::get(ctx, freeDim, freeExtent, /*stride=*/1);
    SmallVector<DimAttr> slotVariant = base;
    slotVariant.push_back(freePiece);
    computeFootprints.push_back(std::move(slotVariant));
    SmallVector<DimAttr> ctVariant = {freePiece};
    ctVariant.append(base.begin(), base.end());
    computeFootprints.push_back(std::move(ctVariant));
    for (size_t p = 0; p < base.size(); ++p) {
      if (!base[p].isReplicate() || base[p].getSize() != freeExtent) continue;
      SmallVector<DimAttr> replaceVariant = base;
      replaceVariant[p] = freePiece;
      computeFootprints.push_back(std::move(replaceVariant));
    }
  };
  addHostVariants(lhsIter, kMatmulDimJ, jExtent);
  addHostVariants(rhsIter, kMatmulDimI, iExtent);

  // Each footprint yields the roll-free placement plus its rolled diagonal
  // variants: a unit-stride k piece rolled by a same-extent unit-stride
  // traversal piece elsewhere in the footprint. Each operand's expansion
  // inherits the roll positionally (for the operand that does not own the
  // partner dim, the partner is the replication piece that subsumed it, and
  // the roll materializes every rotation across its blocks); the multiply
  // stays one op per ciphertext and the k reduction keeps its footprint
  // shape. Two families fall out of the position pair:
  //   - ct-diagonal (k in the ciphertext prefix, partner in the slots): the
  //     ciphertext axis indexes k-diagonals and the k-sum is plain
  //     ciphertext adds;
  //   - slot-diagonal (k in the slots; a slot partner is the classic
  //     Halevi-Shoup diagonal packing, a ciphertext partner the
  //     replicate-then-roll form whose expansion from a compact source is
  //     pure rotations): the rolled slot axis indexes diagonals and the
  //     k-sum stays the slot rotate-and-reduce -- per remaining slot
  //     coordinate the rolled index is a bijection of k, so the tree still
  //     sums k and the result gaps the piece as usual.
  // Both pieces inside the ciphertext prefix is skipped: that only permutes
  // which ciphertext holds what, which conversion already treats as free.
  SmallVector<std::pair<SmallVector<DimAttr>, SmallVector<int64_t>>>
      computeVariants;
  for (SmallVector<DimAttr>& footprint : computeFootprints) {
    const size_t ctPrefixLen = inferCtPrefixLen(footprint, n);
    computeVariants.push_back({footprint, {}});
    for (size_t kPos = 0; kPos < footprint.size(); ++kPos) {
      DimAttr kPiece = footprint[kPos];
      if (kPiece.getDim() != kMatmulDimK || kPiece.getStride() != 1) continue;
      for (size_t partnerPos = 0; partnerPos < footprint.size(); ++partnerPos) {
        if (partnerPos == kPos) continue;
        if (kPos < ctPrefixLen && partnerPos < ctPrefixLen) continue;
        DimAttr partner = footprint[partnerPos];
        if (partner.isGap() || partner.isReplicate() ||
            partner.getDim() == kMatmulDimK || partner.getStride() != 1 ||
            partner.getSize() != kPiece.getSize()) {
          continue;
        }
        computeVariants.push_back(
            {footprint,
             {static_cast<int64_t>(kPos), static_cast<int64_t>(partnerPos)}});
      }
    }
  }

  llvm::DenseSet<Attribute> seenComputeLayouts;
  SmallVector<MatmulPlan> plans;
  for (const auto& [computeDims, rolls] : computeVariants) {
    LayoutAttr computeLayout = makeCheckedLayout(ctx, computeDims, n, rolls);
    if (!computeLayout) continue;
    if (!seenComputeLayouts.insert(computeLayout).second) continue;

    const size_t ctPrefixLen = inferCtPrefixLen(computeDims, n);
    const int64_t numCtCompute = layoutNumCiphertexts(computeLayout);

    SmallVector<DimAttr> expandedLhsDims =
        replaceDimWithReplication(ctx, computeDims, kMatmulDimJ);
    SmallVector<DimAttr> expandedRhsDims =
        replaceDimWithReplication(ctx, computeDims, kMatmulDimI);

    // Sum away k: a slot piece becomes a gap -- the cyclic log-tree
    // rotate-and-reduce leaves the true sum only at the k=0 offset (other
    // offsets hold window sums whose carries cross into the digit above k),
    // so the result makes no claim about those slots. Ciphertext pieces
    // collapse into adds and are dropped.
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
          DimAttr::get(ctx, /*dim=*/-2, piece.getSize(), /*stride=*/1));
    }

    // The roll is positional, so both expansions inherit it piece for piece
    // (subsumption keeps positions; for the operand that does not own the
    // partner dim the roll lands on the replication piece that replaced it).
    // The result drops the rolls: every roll rewrites a k piece, and summing
    // k consumes that piece (a ciphertext k piece is dropped, a slot k piece
    // becomes a gap) -- per remaining coordinate the rolled index is a
    // bijection of k, so the sum over it is the k-sum and the surviving
    // pieces are unrolled.
    MatmulPlan plan;
    plan.computeLayout = computeLayout;
    plan.expandedLhs = makeCheckedLayout(
        ctx, relabelTraversalDims(ctx, expandedLhsDims, iterToLhs), n, rolls);
    plan.expandedRhs = makeCheckedLayout(
        ctx, relabelTraversalDims(ctx, expandedRhsDims, iterToRhs), n, rolls);
    plan.resultLayout = makeCheckedLayout(
        ctx, relabelTraversalDims(ctx, resultDims, iterToResult), n);
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
