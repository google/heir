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
  // A footprint carries the host's own rolls (positions adjusted for the
  // inserted free piece), so a rolled operand hosts placements that keep its
  // diagonal form at zero conversion.
  struct Footprint {
    SmallVector<DimAttr> dims;
    SmallVector<int64_t> seedRolls;
  };
  SmallVector<Footprint> computeFootprints;
  auto addHostVariants = [&](ArrayRef<DimAttr> host,
                             ArrayRef<int64_t> hostRolls, int64_t freeDim,
                             int64_t freeExtent) {
    SmallVector<DimAttr> base(host.begin(), host.end());
    SmallVector<int64_t> baseRolls(hostRolls.begin(), hostRolls.end());
    if (freeExtent == 1) {
      computeFootprints.push_back({base, baseRolls});
      return;
    }
    DimAttr freePiece = DimAttr::get(ctx, freeDim, freeExtent, /*stride=*/1);
    SmallVector<DimAttr> slotVariant = base;
    slotVariant.push_back(freePiece);
    computeFootprints.push_back({std::move(slotVariant), baseRolls});
    SmallVector<DimAttr> ctVariant = {freePiece};
    ctVariant.append(base.begin(), base.end());
    SmallVector<int64_t> shiftedRolls;
    shiftedRolls.reserve(baseRolls.size());
    for (int64_t position : baseRolls) shiftedRolls.push_back(position + 1);
    computeFootprints.push_back(
        {std::move(ctVariant), std::move(shiftedRolls)});
    for (size_t p = 0; p < base.size(); ++p) {
      if (!base[p].isReplicate() || base[p].getSize() != freeExtent) continue;
      SmallVector<DimAttr> replaceVariant = base;
      replaceVariant[p] = freePiece;
      computeFootprints.push_back({std::move(replaceVariant), baseRolls});
    }
  };
  auto layoutRolls = [](LayoutAttr layout) -> SmallVector<int64_t> {
    DenseI64ArrayAttr rolls = layout.getRolls();
    if (!rolls) return {};
    return SmallVector<int64_t>(rolls.asArrayRef().begin(),
                                rolls.asArrayRef().end());
  };
  addHostVariants(lhsIter, layoutRolls(lhs), kMatmulDimJ, jExtent);
  addHostVariants(rhsIter, layoutRolls(rhs), kMatmulDimI, iExtent);

  // Each footprint yields the roll-free placement, the host's own rolls
  // (seeded above), and single-roll decorations on top of either base: a
  // unit-stride traversal piece rolled by a same-extent unit-stride piece
  // elsewhere in the footprint. The from piece picks the family:
  //   - k rolled by an i/j traversal piece: the diagonal operand packings.
  //     A ciphertext-prefix k rolled by a slot piece is the ct-diagonal
  //     family (k summed by plain ciphertext adds); a slot k rolled by a
  //     slot piece is the Halevi-Shoup diagonal packing and by a ciphertext
  //     piece the replicate-then-roll form (k summed by the usual slot
  //     rotate-and-reduce -- per remaining coordinate the rolled index is a
  //     bijection of k, so the tree still sums k).
  //   - i/j rolled by the other free/host traversal dim or by a replication
  //     piece (the free-swap diagonal): the k-sum fixes (i, j) per remaining
  //     address either way, so the roll passes through the kernel and the
  //     RESULT inherits it -- a matmul can produce a diagonal result
  //     directly, which is exactly the operand form a downstream
  //     Halevi-Shoup matmul wants at zero conversion.
  // Never decorated: a non-k piece rolled by k (summing k at a fixed rolled
  // address would mix values of the rolled dim instead of summing k for one
  // (i, j)), and pairs entirely inside the ciphertext prefix (that only
  // permutes which ciphertext holds what, which conversion already treats
  // as free). Each operand's expansion inherits the rolls positionally,
  // except a roll whose from piece the expansion subsumes into replication
  // (the operand does not own that dim, so the roll is a no-op for it); the
  // multiply stays one op per ciphertext and the k reduction keeps its
  // footprint shape.
  SmallVector<std::pair<SmallVector<DimAttr>, SmallVector<int64_t>>>
      computeVariants;
  for (Footprint& footprint : computeFootprints) {
    ArrayRef<DimAttr> dims = footprint.dims;
    const size_t ctPrefixLen = inferCtPrefixLen(footprint.dims, n);

    // The host's rolls, minus reduction-incompatible pairs (non-k rolled by
    // k). Positions relabel one to one, so a compatible-rolled host is the
    // zero-conversion operand of the variants it seeds.
    SmallVector<int64_t> seed;
    for (size_t i = 0; i + 1 < footprint.seedRolls.size(); i += 2) {
      DimAttr from = dims[footprint.seedRolls[i]];
      DimAttr by = dims[footprint.seedRolls[i + 1]];
      if (from.getDim() != kMatmulDimK && by.getDim() == kMatmulDimK) continue;
      seed.push_back(footprint.seedRolls[i]);
      seed.push_back(footprint.seedRolls[i + 1]);
    }

    SmallVector<std::pair<int64_t, int64_t>> decorations;
    for (size_t fromPos = 0; fromPos < dims.size(); ++fromPos) {
      DimAttr fromPiece = dims[fromPos];
      if (fromPiece.isGap() || fromPiece.isReplicate() ||
          fromPiece.getStride() != 1) {
        continue;
      }
      for (size_t byPos = 0; byPos < dims.size(); ++byPos) {
        if (byPos == fromPos) continue;
        if (fromPos < ctPrefixLen && byPos < ctPrefixLen) continue;
        DimAttr byPiece = dims[byPos];
        if (byPiece.isGap() || byPiece.getStride() != 1 ||
            byPiece.getSize() != fromPiece.getSize() ||
            byPiece.getDim() == kMatmulDimK) {
          continue;
        }
        if (fromPiece.getDim() == kMatmulDimK) {
          // The diagonal operand families roll k by a traversal partner; a
          // replication partner is the free-swap form below, meaningless
          // for k (its rotations are what the reduction consumes).
          if (byPiece.isReplicate()) continue;
        } else {
          // Free-swap family: i/j by the other traversal dim or by
          // replication; rolling a dim by another piece of itself adds
          // nothing a conversion doesn't.
          if (!byPiece.isReplicate() &&
              byPiece.getDim() == fromPiece.getDim()) {
            continue;
          }
        }
        decorations.push_back(
            {static_cast<int64_t>(fromPos), static_cast<int64_t>(byPos)});
      }
    }

    SmallVector<SmallVector<int64_t>> bases;
    bases.push_back({});
    if (!seed.empty()) bases.push_back(seed);
    for (const SmallVector<int64_t>& base : bases) {
      computeVariants.push_back({footprint.dims, base});
      for (auto [fromPos, byPos] : decorations) {
        // No two rolls may rewrite the same piece, and no roll may involve
        // a piece another roll rewrites (composition would be
        // order-dependent); such stacks are left to future variants.
        bool conflicts = false;
        for (size_t i = 0; i + 1 < base.size(); i += 2) {
          if (base[i] == fromPos || base[i] == byPos ||
              base[i + 1] == fromPos) {
            conflicts = true;
            break;
          }
        }
        if (conflicts) continue;
        SmallVector<int64_t> combined = base;
        combined.push_back(fromPos);
        combined.push_back(byPos);
        computeVariants.push_back({footprint.dims, std::move(combined)});
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
    SmallVector<int64_t> resultPosition(computeDims.size(), -1);
    int64_t kSlotExtent = 1;
    int64_t kCtExtent = 1;
    for (size_t p = 0; p < computeDims.size(); ++p) {
      DimAttr piece = computeDims[p];
      if (piece.getDim() != kMatmulDimK) {
        resultPosition[p] = static_cast<int64_t>(resultDims.size());
        resultDims.push_back(piece);
        continue;
      }
      if (p < ctPrefixLen) {
        kCtExtent *= piece.getSize();
        continue;
      }
      kSlotExtent *= piece.getSize();
      resultPosition[p] = static_cast<int64_t>(resultDims.size());
      resultDims.push_back(
          DimAttr::get(ctx, /*dim=*/-2, piece.getSize(), /*stride=*/1));
    }

    // The rolls are positional, so both expansions inherit them piece for
    // piece (subsumption keeps positions; for the operand that does not own
    // a partner dim the roll lands on the replication piece that replaced
    // it, materializing every rotation across its blocks) -- except a roll
    // whose FROM piece the expansion subsumes: the operand does not own
    // that dim, so its placement is plain replication there and the roll is
    // dropped. The result drops the k rolls (summing k consumes the rolled
    // piece; per remaining coordinate the rolled index is a bijection of k,
    // so the sum over it is still the k-sum) and keeps the i/j rolls at
    // their surviving positions -- a diagonal result.
    auto rollsWithoutFromDim = [&](int64_t subsumedDim) {
      SmallVector<int64_t> kept;
      for (size_t i = 0; i + 1 < rolls.size(); i += 2) {
        if (computeDims[rolls[i]].getDim() == subsumedDim) continue;
        kept.push_back(rolls[i]);
        kept.push_back(rolls[i + 1]);
      }
      return kept;
    };
    SmallVector<int64_t> resultRolls;
    for (size_t i = 0; i + 1 < rolls.size(); i += 2) {
      if (computeDims[rolls[i]].getDim() == kMatmulDimK) continue;
      const int64_t from = resultPosition[rolls[i]];
      const int64_t by = resultPosition[rolls[i + 1]];
      assert(from >= 0 && by >= 0 &&
             "non-k roll pieces must survive the k-summation");
      resultRolls.push_back(from);
      resultRolls.push_back(by);
    }

    MatmulPlan plan;
    plan.computeLayout = computeLayout;
    plan.expandedLhs = makeCheckedLayout(
        ctx, relabelTraversalDims(ctx, expandedLhsDims, iterToLhs), n,
        rollsWithoutFromDim(kMatmulDimJ));
    plan.expandedRhs = makeCheckedLayout(
        ctx, relabelTraversalDims(ctx, expandedRhsDims, iterToRhs), n,
        rollsWithoutFromDim(kMatmulDimI));
    plan.resultLayout = makeCheckedLayout(
        ctx, relabelTraversalDims(ctx, resultDims, iterToResult), n,
        resultRolls);
    if (!plan.expandedLhs || !plan.expandedRhs || !plan.resultLayout) continue;
    // The result becomes a value's assigned layout, so it must actually
    // materialize (the expansions are only priced, and unmaterializable
    // ones are already rejected there).
    if (!isMaterializableRotomLayout(plan.resultLayout)) continue;

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
