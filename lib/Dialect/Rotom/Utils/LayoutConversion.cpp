#include "lib/Dialect/Rotom/Utils/LayoutConversion.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomLayout.h"
#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

namespace {
// Where a traversal piece lands: in the ciphertext dims or the slot dims,
// and at what offset. The address is the sum of each piece's dims times its
// offset.
struct PiecePlacement {
  bool inSlots;
  // The product of the extents of the pieces after this one in its region.
  // Ciphertext-side offsets count too, so a reorder on the ciphertext side is
  // free only when the piece stays on that side.
  int64_t offset;
  int64_t extent;
};
}  // namespace

// Computes the placement of every traversal piece, keyed by (dim, stride).
// The stride key is 1 for a dim with a single piece. Also returns the product
// of the slot-side replication extents in `slotReplication`.
static llvm::DenseMap<std::pair<int64_t, int64_t>, PiecePlacement>
piecePlacements(LayoutAttr layout, int64_t& slotReplication) {
  SmallVector<DimAttr> dims;
  for (Attribute attr : layout.getDims()) dims.push_back(cast<DimAttr>(attr));
  const size_t ctPrefixLen = inferCtPrefixLen(dims, layout.getN());

  llvm::DenseMap<int64_t, int64_t> pieceCount;
  for (DimAttr piece : dims) {
    if (!piece.isGap() && !piece.isReplicate()) ++pieceCount[piece.getDim()];
  }

  // Suffix offsets within each region, right to left.
  llvm::DenseMap<std::pair<int64_t, int64_t>, PiecePlacement> placements;
  slotReplication = 1;
  int64_t offset = 1;
  for (size_t p = dims.size(); p-- > 0;) {
    if (p + 1 == ctPrefixLen) offset = 1;  // entering the ciphertext region
    DimAttr piece = dims[p];
    const bool inSlots = p >= ctPrefixLen;
    if (piece.isReplicate() || piece.isGap()) {
      // Only replication counts as a fill: a gap's blocks hold nothing, so
      // producing them is free. A gap is NOT free for the pieces to its
      // left.
      if (inSlots && piece.isReplicate() && piece.getSize() > 1) {
        slotReplication *= piece.getSize();
      }
      offset *= piece.getSize();
      continue;
    }
    if (piece.getSize() > 1) {
      const int64_t strideKey =
          pieceCount[piece.getDim()] > 1 ? piece.getStride() : 1;
      placements[{piece.getDim(), strideKey}] =
          PiecePlacement{inSlots, offset, piece.getSize()};
    }
    offset *= piece.getSize();
  }
  return placements;
}

#include "llvm/include/llvm/ADT/DenseSet.h"  // from @llvm-project
// Shared engine for the address-only conversion and the one-roll alignment.
namespace {
struct PositionedPiece {
  int64_t pos;  // position in the dims list
  int64_t dim;  // axis, or the R/G sentinel
  int64_t size;
  int64_t divisor;  // the piece's stride: index = (axis index / divisor) % size
  int64_t offset;   // address offset within its region
  bool inSlots;
  bool isRepl() const { return dim == -1; }
  bool isGap() const { return dim == -2; }
};

SmallVector<PositionedPiece> positionedPieces(LayoutAttr layout) {
  SmallVector<DimAttr> dims = layoutDims(layout);
  const size_t ctLen = inferCtPrefixLen(dims, layout.getN());
  SmallVector<PositionedPiece> out(dims.size());
  int64_t offset = 1;
  for (size_t p = dims.size(); p-- > 0;) {
    if (p + 1 == ctLen) offset = 1;
    DimAttr d = dims[p];
    out[p] = PositionedPiece{static_cast<int64_t>(p), d.getDim(), d.getSize(),
                             d.getStride(),           offset,     p >= ctLen};
    offset *= d.getSize();
  }
  return out;
}

// An axis this conversion has to move: its pieces on each side. The axis id
// and its extent both follow from `dst`.
struct MovingAxis {
  SmallVector<PositionedPiece> src, dst;
  int64_t dim() const { return dst.front().dim; }
  int64_t extent() const {
    int64_t e = 1;
    for (const PositionedPiece& p : dst) e *= p.size;
    return e;
  }
};
}  // namespace

namespace {
// A roll names its arguments by piece position, so two layouts carry the same
// rolls when their roll lists match position for position.
bool sameRollRelation(ArrayRef<RollSpec> fromRolls,
                      ArrayRef<RollSpec> toRolls) {
  if (fromRolls.size() != toRolls.size()) return false;
  for (size_t i = 0; i < fromRolls.size(); ++i) {
    if (!(fromRolls[i].from == toRolls[i].from &&
          fromRolls[i].by == toRolls[i].by)) {
      return false;
    }
  }
  return true;
}
}  // namespace

FailureOr<ConversionPlan> planLayoutConversion(LayoutAttr from, LayoutAttr to) {
  if (!from || !to || from.getN() != to.getN()) return failure();
  const int64_t n = from.getN();
  SmallVector<RollSpec> fromRolls = getRollSpecs(from);
  SmallVector<RollSpec> toRolls = getRollSpecs(to);
  int64_t rollFromPos = -1, rollByPos = -1;
  if (toRolls.size() == fromRolls.size() + 1) {
    const RollSpec& added = toRolls.back();
    if (added.from.isAxis || added.by.isAxis) return failure();  // TODO: axis
    rollFromPos = added.from.index;
    rollByPos = added.by.index;
    toRolls.pop_back();
  } else if (toRolls.size() != fromRolls.size()) {
    return failure();
  }
  // The rolls both sides share must be the same relation; a different roll
  // is a content change that no address plan expresses.
  if (!sameRollRelation(fromRolls, toRolls)) return failure();

  SmallVector<PositionedPiece> fromP = positionedPieces(from);
  SmallVector<PositionedPiece> toP = positionedPieces(to);
  if (rollFromPos >= static_cast<int64_t>(toP.size()) ||
      rollByPos >= static_cast<int64_t>(toP.size())) {
    return failure();
  }
  // Below every dim id, including the R/G/K sentinels.
  constexpr int64_t kNoAxis = std::numeric_limits<int64_t>::min();
  const int64_t rollAxis = rollFromPos >= 0 ? toP[rollFromPos].dim : kNoAxis;
  const bool byIsRepl = rollByPos >= 0 && toP[rollByPos].isRepl();
  const int64_t byAxis =
      (rollByPos >= 0 && !byIsRepl) ? toP[rollByPos].dim : kNoAxis;

  // Group traversal pieces by axis.
  auto groupAxes = [](ArrayRef<PositionedPiece> pieces) {
    std::map<int64_t, SmallVector<PositionedPiece>> axes;
    for (const PositionedPiece& p : pieces) {
      if (p.isRepl() || p.isGap()) continue;
      axes[p.dim].push_back(p);
    }
    return axes;
  };
  auto fromAxes = groupAxes(fromP);
  auto toAxes = groupAxes(toP);
  if (fromAxes.size() != toAxes.size()) return failure();

  auto extentOf = [](ArrayRef<PositionedPiece> ps) {
    int64_t e = 1;
    for (const PositionedPiece& p : ps) e *= p.size;
    return e;
  };
  auto shape = [](ArrayRef<PositionedPiece> ps) {
    SmallVector<std::tuple<int64_t, int64_t, int64_t, bool>> s;
    for (const PositionedPiece& p : ps) {
      s.push_back({p.divisor, p.size, p.offset, p.inSlots});
    }
    llvm::sort(s);
    return s;
  };

  SmallVector<MovingAxis> moving;
  SmallVector<std::pair<int64_t, int64_t>> fixedSpans;  // (offset, extent)
  for (auto& [dim, dstPieces] : toAxes) {
    auto it = fromAxes.find(dim);
    if (it == fromAxes.end()) return failure();
    const SmallVector<PositionedPiece>& srcPieces = it->second;
    const int64_t extent = extentOf(dstPieces);
    if (extentOf(srcPieces) != extent) return failure();
    bool allSlots = true;
    for (const PositionedPiece& p : dstPieces) allSlots &= p.inSlots;
    const bool same = allSlots && shape(srcPieces) == shape(dstPieces) &&
                      dim != rollAxis && dim != byAxis;
    if (same) {
      for (const PositionedPiece& p : dstPieces) {
        fixedSpans.push_back({p.offset, p.size});
      }
      continue;
    }
    moving.push_back({srcPieces, dstPieces});
  }

  // Replication: destination replicas are targets to write, source replicas a
  // free choice of which copy to read.
  SmallVector<PositionedPiece> dstReplicas, srcReplicas;
  SmallVector<ReplicationFill> fills;
  int byReplicaIdx = -1;
  // Slot replication the source already holds, by (offset, extent). A cyclic
  // rotation keeps a period, so a step never disturbs it: the target gets it
  // for nothing, and filling it again would double every target ciphertext
  // log2(E) times over.
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 4> srcSlotRepl;
  for (const PositionedPiece& p : fromP) {
    if (p.isRepl() && p.size > 1 && p.inSlots) {
      srcSlotRepl.insert({p.offset, p.size});
    }
  }
  for (const PositionedPiece& p : toP) {
    if (!p.isRepl() || p.size <= 1) continue;
    if (p.inSlots && p.pos != rollByPos &&
        srcSlotRepl.contains({p.offset, p.size})) {
      // Already there, so neither a fill nor a coordinate -- but the step
      // still writes every copy: a rotated replicated row is correct in all
      // of its blocks, so the step covers them and needs no mask.
      fixedSpans.push_back({p.offset, p.size});
      continue;
    }
    // A slot-side replication is filled by doubling once one copy is in
    // place, so it is not a target coordinate: enumerating it would cost one
    // rotation per copy of every value. The roll's BY argument stays a
    // coordinate, because the roll reads its index.
    if (p.inSlots && p.pos != rollByPos) {
      fills.push_back({p.offset, p.size});
      continue;
    }
    if (p.pos == rollByPos) byReplicaIdx = dstReplicas.size();
    dstReplicas.push_back(p);
  }
  if (byIsRepl && byReplicaIdx < 0) return failure();
  llvm::sort(fixedSpans);
  for (const PositionedPiece& p : fromP) {
    if (!p.isRepl() || p.size <= 1) continue;
    srcReplicas.push_back(p);
  }

  // The slots a step writes: the target slot plus every combination of the
  // fixed axes' pieces.
  SmallVector<int64_t> fixedOffsets = {0};
  for (auto [offset, extent] : fixedSpans) {
    SmallVector<int64_t> next;
    next.reserve(fixedOffsets.size() * extent);
    for (int64_t off : fixedOffsets) {
      for (int64_t d = 0; d < extent; ++d) next.push_back(off + d * offset);
    }
    fixedOffsets = std::move(next);
  }
  llvm::sort(fixedOffsets);

  int64_t targetCombos = 1, replicaCombos = 1;
  for (const MovingAxis& a : moving) targetCombos *= a.extent();
  for (const PositionedPiece& r : dstReplicas) targetCombos *= r.size;
  for (const PositionedPiece& r : srcReplicas) replicaCombos *= r.size;

  std::map<std::tuple<int64_t, int64_t, int64_t>, SmallVector<int64_t>> groups;
  SmallVector<int64_t> axisValue(moving.size(), 0);
  SmallVector<int64_t> dstReplValue(dstReplicas.size(), 0);
  SmallVector<int64_t> srcReplValue(srcReplicas.size(), 0);
  auto indexOf = [](const PositionedPiece& p, int64_t axisIndex) {
    return (axisIndex / p.divisor) % p.size;
  };
  for (int64_t tc = 0; tc < targetCombos; ++tc) {
    int64_t rest = tc;
    for (size_t i = 0; i < moving.size(); ++i) {
      axisValue[i] = rest % moving[i].extent();
      rest /= moving[i].extent();
    }
    for (size_t i = 0; i < dstReplicas.size(); ++i) {
      dstReplValue[i] = rest % dstReplicas[i].size;
      rest /= dstReplicas[i].size;
    }
    // The index the applied roll subtracts, if any.
    int64_t byValue = 0;
    if (rollByPos >= 0) {
      if (byIsRepl) {
        byValue = dstReplValue[byReplicaIdx];
      } else {
        for (size_t i = 0; i < moving.size(); ++i) {
          if (moving[i].dim() == byAxis) {
            byValue = indexOf(toP[rollByPos], axisValue[i]);
          }
        }
      }
    }
    int64_t targetCt = 0, targetSlot = 0;
    for (size_t i = 0; i < moving.size(); ++i) {
      for (const PositionedPiece& p : moving[i].dst) {
        int64_t index = indexOf(p, axisValue[i]);
        if (p.pos == rollFromPos) {
          index = ((index - byValue) % p.size + p.size) % p.size;
        }
        (p.inSlots ? targetSlot : targetCt) += index * p.offset;
      }
    }
    for (size_t i = 0; i < dstReplicas.size(); ++i) {
      (dstReplicas[i].inSlots ? targetSlot : targetCt) +=
          dstReplValue[i] * dstReplicas[i].offset;
    }

    std::optional<std::tuple<int64_t, int64_t, int64_t>> best;
    bool bestExisting = false;
    for (int64_t rc = 0; rc < replicaCombos; ++rc) {
      int64_t r = rc;
      for (size_t i = 0; i < srcReplicas.size(); ++i) {
        srcReplValue[i] = r % srcReplicas[i].size;
        r /= srcReplicas[i].size;
      }
      int64_t sourceCt = 0, sourceSlot = 0;
      for (size_t i = 0; i < moving.size(); ++i) {
        for (const PositionedPiece& p : moving[i].src) {
          (p.inSlots ? sourceSlot : sourceCt) +=
              indexOf(p, axisValue[i]) * p.offset;
        }
      }
      for (size_t i = 0; i < srcReplicas.size(); ++i) {
        (srcReplicas[i].inSlots ? sourceSlot : sourceCt) +=
            srcReplValue[i] * srcReplicas[i].offset;
      }
      const int64_t shift = ((sourceSlot - targetSlot) % n + n) % n;
      std::tuple<int64_t, int64_t, int64_t> key{targetCt, sourceCt, shift};
      const bool existing = groups.count(key) > 0;
      auto rank = [](bool existing, const auto& key) {
        return std::tuple(!existing, std::get<2>(key) != 0, key);
      };
      if (!best || rank(existing, key) < rank(bestExisting, *best)) {
        best = key;
        bestExisting = existing;
      }
    }
    auto& slots = groups[*best];
    for (int64_t off : fixedOffsets) slots.push_back(targetSlot + off);
  }

  SmallVector<LayoutConversionStep> steps;
  steps.reserve(groups.size());
  for (auto& [key, slots] : groups) {
    auto [targetCt, sourceCt, shift] = key;
    llvm::sort(slots);
    slots.erase(llvm::unique(slots), slots.end());
    steps.push_back(
        LayoutConversionStep{targetCt, sourceCt, shift, std::move(slots)});
  }
  llvm::sort(fills, [](const ReplicationFill& x, const ReplicationFill& y) {
    return std::tuple(x.stride, x.extent) < std::tuple(y.stride, y.extent);
  });
  return ConversionPlan{std::move(steps), std::move(fills)};
}

ConversionEstimate estimateConversionCost(LayoutAttr from, LayoutAttr to) {
  ConversionEstimate estimate;
  if (!from || !to) return estimate;
  // Layouts with the same merged canonical form are the same packing, so the
  // conversion is free.
  if (from == to ||
      mergeAdjacentLayoutDims(from) == mergeAdjacentLayoutDims(to)) {
    return estimate;
  }

  // The price is the plan. Anything else lets the search choose a conversion
  // the lowering cannot emit, or one that costs orders of magnitude more than
  // it was charged.
  FailureOr<ConversionPlan> plan = planLayoutConversion(from, to);
  if (failed(plan)) {
    estimate.lowerable = false;
    return estimate;
  }

  const int64_t n = to.getN();
  // One rotation per distinct (source ciphertext, shift): the emission reuses
  // a rotated row across every target it feeds.
  llvm::DenseSet<std::pair<int64_t, int64_t>> rotations;
  llvm::DenseSet<int64_t> written;
  for (const LayoutConversionStep& step : plan->steps) {
    if (step.shift != 0) rotations.insert({step.sourceCt, step.shift});
    if (static_cast<int64_t>(step.targetSlots.size()) != n) ++estimate.masks;
    // The first step of a target ciphertext writes it; the rest add into it.
    if (!written.insert(step.targetCt).second) ++estimate.accumulates;
  }
  estimate.rotations = rotations.size();
  // A fill doubles: log2(extent) rotate-and-add steps, whatever the extent.
  for (const ReplicationFill& fill : plan->fills) {
    const int64_t doublings = llvm::Log2_64_Ceil(fill.extent);
    estimate.rotations += doublings;
    estimate.accumulates += doublings;
  }
  return estimate;
}

std::optional<BsgsSchedule> bsgsScheduleOpt(LayoutAttr from, LayoutAttr to) {
  if (!from || !to || from.getN() != to.getN()) return std::nullopt;
  const int64_t n = to.getN();
  SmallVector<RollSpec> fromRolls = getRollSpecs(from);
  SmallVector<RollSpec> toRolls = getRollSpecs(to);
  // `to` is `from` plus exactly one roll, the one alignment just built.
  if (toRolls.size() != fromRolls.size() + 1) return std::nullopt;
  for (size_t i = 0; i < fromRolls.size(); ++i) {
    if (!(fromRolls[i].from == toRolls[i].from &&
          fromRolls[i].by == toRolls[i].by)) {
      return std::nullopt;
    }
  }
  const RollSpec roll = toRolls.back();
  if (roll.from.isAxis || roll.by.isAxis) return std::nullopt;

  // The fold reads one source ciphertext, so the source must hold identical
  // rows: nothing but replication and gaps addresses its ciphertexts.
  SmallVector<DimAttr> fromDims = layoutDims(from);
  const size_t fromCtLen = inferCtPrefixLen(fromDims, n);
  for (size_t p = 0; p < fromCtLen; ++p) {
    if (!fromDims[p].isReplicate() && !fromDims[p].isGap()) return std::nullopt;
  }

  SmallVector<DimAttr> toDims = layoutDims(to);
  const size_t toCtLen = inferCtPrefixLen(toDims, n);
  const int64_t rolled = roll.from.index;
  const int64_t by = roll.by.index;
  if (rolled < 0 || by < 0 || static_cast<size_t>(rolled) >= toCtLen ||
      static_cast<size_t>(by) < toCtLen ||
      static_cast<size_t>(by) >= toDims.size()) {
    return std::nullopt;
  }
  // The ciphertext piece traversing the targets, rolled by the replication it
  // traded places with: same extent, or the shifts would not close.
  const int64_t targets = toDims[rolled].getSize();
  if (targets < 2 || toDims[rolled].isReplicate() || toDims[rolled].isGap()) {
    return std::nullopt;
  }
  if (!toDims[by].isReplicate() || toDims[by].getSize() != targets) {
    return std::nullopt;
  }
  // A piece's slot offset is the product of the extents to its right.
  int64_t stride = 1;
  for (size_t p = by + 1; p < toDims.size(); ++p) {
    stride *= toDims[p].getSize();
  }
  if (stride <= 0 || stride >= n) return std::nullopt;
  return BsgsSchedule{stride, targets, /*negative=*/false};
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
