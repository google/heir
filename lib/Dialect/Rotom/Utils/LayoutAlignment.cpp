#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomLayout.h"
#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {
OperatorAlignmentMap OperatorAlignmentMap::identity(int64_t rank) {
  OperatorAlignmentMap map;
  for (int64_t d = 0; d < rank; ++d) {
    map.lhsToRhs.push_back(d);
    map.rhsToLhs.push_back(d);
  }
  return map;
}

OperatorAlignmentMap OperatorAlignmentMap::matmul(int64_t lhsRank,
                                                  int64_t rhsRank) {
  assert(lhsRank >= 1 && rhsRank >= 1 && "matmul operands need rank >= 1");
  // A rank-1 operand is [k]. Otherwise the lhs is [..batch, m, k] and the
  // rhs is [..batch, k, n].
  const int64_t lhsBatch = std::max<int64_t>(lhsRank - 2, 0);
  const int64_t rhsBatch = std::max<int64_t>(rhsRank - 2, 0);
  const int64_t lhsK = lhsRank - 1;
  const int64_t rhsK = rhsRank == 1 ? 0 : rhsRank - 2;

  OperatorAlignmentMap map;
  map.lhsToRhs.assign(lhsRank, kRepeatedDim);
  map.rhsToLhs.assign(rhsRank, kRepeatedDim);
  // Batch dims pair from the right.
  const int64_t shared = std::min(lhsBatch, rhsBatch);
  for (int64_t b = 0; b < shared; ++b) {
    const int64_t l = lhsBatch - 1 - b;
    const int64_t r = rhsBatch - 1 - b;
    map.lhsToRhs[l] = r;
    map.rhsToLhs[r] = l;
  }
  // The aligned dims pair with each other.
  map.lhsToRhs[lhsK] = rhsK;
  map.rhsToLhs[rhsK] = lhsK;
  return map;
}

namespace {

// Helper function to check traversal dimension alignment. For each dimension,
// check if the size, stride, and type (gap or replicate) match according
// to the operator alignment map.
SmallVector<DimAttr> piecesOf(ArrayAttr dims,
                              SmallVector<int64_t>* posMap = nullptr) {
  SmallVector<DimAttr> out;
  if (posMap) posMap->clear();
  for (Attribute attr : dims) {
    auto d = cast<DimAttr>(attr);
    if (posMap) posMap->push_back(out.size());
    out.push_back(d);
  }
  return out;
}

bool checkDimAlignment(const OperatorAlignmentMap& map,
                       ArrayRef<DimAttr> lhsDims, ArrayRef<DimAttr> rhsDims) {
  // Check if the dimensions are aligned.
  // The reference uses two pointers to traverse the layout dimensions.
  // The side whose run is smaller advances until the two runs hold the
  // same extent, so one piece may face several on the other side: e.g.,
  // [R:64] against [R:4][i:16].
  auto runExtent = [](ArrayRef<DimAttr> dims, size_t from, size_t to) {
    int64_t e = 1;
    for (size_t i = from; i <= to; ++i) e *= dims[i].getSize();
    return e;
  };
  auto isFill = [](DimAttr d) { return !d.isGap(); };
  size_t aStart = 0, bStart = 0, aNext = 0, bNext = 0;
  while (aNext < lhsDims.size() && bNext < rhsDims.size()) {
    DimAttr a = lhsDims[aNext];
    DimAttr b = rhsDims[bNext];
    // Two gaps of one extent match outright.
    if (a.isGap() && b.isGap() && a.getSize() == b.getSize()) {
      ++aNext;
      ++bNext;
      aStart = aNext;
      bStart = bNext;
      continue;
    }
    // A gap never faces data.
    if (isFill(a) != isFill(b)) return false;
    // Two pieces that both name no axis (replication, or gaps of different
    // extents) only need their runs to agree.
    const bool aNamesAxis = isFill(a) && !a.isReplicate();
    const bool bNamesAxis = isFill(b) && !b.isReplicate();
    if (aNamesAxis || bNamesAxis) {
      // The pair must be aligned by the operator map.
      bool aligned;
      if (aNamesAxis && bNamesAxis) {
        aligned = map.lhsToRhs[a.getDim()] == b.getDim();
      } else if (aNamesAxis) {
        aligned =
            map.lhsToRhs[a.getDim()] == OperatorAlignmentMap::kRepeatedDim;
      } else {
        aligned =
            map.rhsToLhs[b.getDim()] == OperatorAlignmentMap::kRepeatedDim;
      }
      if (!aligned) return false;
      // A piece facing exactly one piece keeps the stride rule.
      if (aNamesAxis && bNamesAxis && aStart == aNext && bStart == bNext &&
          a.getSize() == b.getSize() && a.getStride() != b.getStride()) {
        return false;
      }
    }
    const int64_t aExtent = runExtent(lhsDims, aStart, aNext);
    const int64_t bExtent = runExtent(rhsDims, bStart, bNext);
    if (aExtent == bExtent) {
      ++aNext;
      ++bNext;
      aStart = aNext;
      bStart = bNext;
    } else if (aExtent > bExtent) {
      ++bNext;
    } else {
      ++aNext;
    }
  }
  return aNext == lhsDims.size() && bNext == rhsDims.size();
}

// Determines if a roll is exempt from alignment. A roll is exempt when the
// other side has replication aligned to the roll's FROM piece.
bool rollExempt(const OperatorAlignmentMap& map, const RollArg& from,
                bool fromLhs, ArrayAttr lhsDims, ArrayAttr rhsDims) {
  if (from.isAxis) {
    const SmallVector<int64_t>& forward = fromLhs ? map.lhsToRhs : map.rhsToLhs;
    return forward[from.index] == OperatorAlignmentMap::kRepeatedDim;
  }
  ArrayAttr other = fromLhs ? rhsDims : lhsDims;
  return cast<DimAttr>(other[from.index]).isReplicate();
}

// Determines the rolls that are required for alignment.
SmallVector<RollSpec> requiredRolls(const OperatorAlignmentMap& map,
                                    LayoutAttr merged, bool isLhs,
                                    ArrayAttr lhsDims, ArrayAttr rhsDims) {
  SmallVector<RollSpec> out;
  for (const RollSpec& roll : getRollSpecs(merged)) {
    ArrayAttr own = isLhs ? lhsDims : rhsDims;
    if (!rollExempt(map, roll.from, isLhs, lhsDims, rhsDims))
      out.push_back(roll);
  }
  return out;
}

// Helper function to check if two roll arguments are the same.
bool sameRollArg(const RollArg& a, const RollArg& b) {
  return a.isAxis == b.isAxis && a.index == b.index;
}

// Helper function to check if the required rolls on one side have the same
// roll on the other side.
bool checkRollAlignment(const OperatorAlignmentMap& map, LayoutAttr lhsMerged,
                        LayoutAttr rhsMerged, ArrayAttr lhsDims,
                        ArrayAttr rhsDims) {
  SmallVector<RollSpec> lhsRolls =
      requiredRolls(map, lhsMerged, /*isLhs=*/true, lhsDims, rhsDims);
  SmallVector<RollSpec> rhsRolls =
      requiredRolls(map, rhsMerged, /*isLhs=*/false, lhsDims, rhsDims);
  if (lhsRolls.size() != rhsRolls.size()) return false;
  SmallVector<int64_t> lhsPos, rhsPos;
  piecesOf(lhsDims, &lhsPos);
  piecesOf(rhsDims, &rhsPos);
  auto argsCorrespond = [&](const RollArg& a, const RollArg& b) {
    if (a.isAxis != b.isAxis) return false;
    if (a.isAxis) return map.lhsToRhs[a.index] == b.index;
    return lhsPos[a.index] == rhsPos[b.index];
  };
  for (size_t i = 0; i < lhsRolls.size(); ++i) {
    if (!argsCorrespond(lhsRolls[i].from, rhsRolls[i].from) ||
        !argsCorrespond(lhsRolls[i].by, rhsRolls[i].by)) {
      return false;
    }
  }
  return true;
}
}  // namespace

// Helper function to check if two layouts are aligned.
bool isOperatorAligned(const OperatorAlignmentMap& map, LayoutAttr lhs,
                       LayoutAttr rhs) {
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return false;
  LayoutAttr lhsMerged = mergeAdjacentLayoutDims(lhs);
  LayoutAttr rhsMerged = mergeAdjacentLayoutDims(rhs);
  ArrayAttr lhsDims = lhsMerged.getDims();
  ArrayAttr rhsDims = rhsMerged.getDims();
  SmallVector<DimAttr> lhsNU = piecesOf(lhsDims);
  SmallVector<DimAttr> rhsNU = piecesOf(rhsDims);
  if (!checkDimAlignment(map, lhsNU, rhsNU)) return false;

  return checkRollAlignment(map, lhsMerged, rhsMerged, lhsDims, rhsDims);
}

// Helper function to align the rolls of two layouts.
FailureOr<RollAlignment> alignRolls(const OperatorAlignmentMap& map,
                                    LayoutAttr lhs, LayoutAttr rhs) {
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return failure();
  LayoutAttr lhsMerged = mergeAdjacentLayoutDims(lhs);
  LayoutAttr rhsMerged = mergeAdjacentLayoutDims(rhs);
  ArrayAttr lhsDims = lhsMerged.getDims();
  ArrayAttr rhsDims = rhsMerged.getDims();
  SmallVector<DimAttr> lhsNU = piecesOf(lhsDims);
  SmallVector<DimAttr> rhsNU = piecesOf(rhsDims);
  if (lhsNU.size() != rhsNU.size()) return failure();
  if (!checkDimAlignment(map, lhsNU, rhsNU)) return failure();

  SmallVector<RollSpec> lhsRolls =
      requiredRolls(map, lhsMerged, /*isLhs=*/true, lhsDims, rhsDims);
  SmallVector<RollSpec> rhsRolls =
      requiredRolls(map, rhsMerged, /*isLhs=*/false, lhsDims, rhsDims);

  // Translate a roll from one layout to the other.
  auto translate = [&](const RollArg& arg,
                       bool fromLhs) -> std::optional<RollArg> {
    if (!arg.isAxis) return arg;
    const SmallVector<int64_t>& forward = fromLhs ? map.lhsToRhs : map.rhsToLhs;
    if (arg.index < 0 || arg.index >= static_cast<int64_t>(forward.size())) {
      return std::nullopt;
    }
    const int64_t mapped = forward[arg.index];
    if (mapped == OperatorAlignmentMap::kRepeatedDim) return std::nullopt;
    return RollArg{/*isAxis=*/true, mapped};
  };
  auto contains = [](ArrayRef<RollSpec> rolls, const RollSpec& want) {
    for (const RollSpec& have : rolls) {
      if (sameRollArg(have.from, want.from) && sameRollArg(have.by, want.by)) {
        return true;
      }
    }
    return false;
  };

  RollAlignment out;
  for (const RollSpec& roll : lhsRolls) {
    std::optional<RollArg> from = translate(roll.from, /*fromLhs=*/true);
    std::optional<RollArg> by = translate(roll.by, /*fromLhs=*/true);
    if (!from || !by) return failure();
    RollSpec translated{*from, *by};
    if (!contains(rhsRolls, translated)) out.addToRhs.push_back(translated);
  }
  for (const RollSpec& roll : rhsRolls) {
    std::optional<RollArg> from = translate(roll.from, /*fromLhs=*/false);
    std::optional<RollArg> by = translate(roll.by, /*fromLhs=*/false);
    if (!from || !by) return failure();
    RollSpec translated{*from, *by};
    if (!contains(lhsRolls, translated)) out.addToLhs.push_back(translated);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Alignment steps
// ---------------------------------------------------------------------------

namespace {
DimAttr mk(MLIRContext* ctx, int64_t dim, int64_t size, int64_t stride) {
  return DimAttr::get(ctx, dim, size, stride);
}
bool isRepl(DimAttr d) { return d.isReplicate(); }
bool isTraversal(DimAttr d) { return !d.isGap() && !d.isReplicate(); }

// Computes the product of every non-gap extent (the reference's FILL
// length).
int64_t fillLen(ArrayRef<DimAttr> dims) {
  int64_t out = 1;
  for (DimAttr d : dims) {
    if (!d.isGap()) out *= d.getSize();
  }
  return out;
}
int64_t capacity(ArrayRef<DimAttr> dims) {
  int64_t out = 1;
  for (DimAttr d : dims) out *= d.getSize();
  return out;
}

// Determines the dims of one side that the map aligns with replication on
// the other side, in ascending order.
SmallVector<int64_t> broadcastDims(const SmallVector<int64_t>& otherToSide) {
  SmallVector<int64_t> out;
  for (int64_t d = 0; d < static_cast<int64_t>(otherToSide.size()); ++d) {
    if (otherToSide[d] == OperatorAlignmentMap::kRepeatedDim) out.push_back(d);
  }
  return out;
}

// Helper function for alignedDims:
// - builds a piece list for `dst`'s tensor in `src`'s piece order.
std::optional<SmallVector<DimAttr>> mirrorDims(
    ArrayRef<DimAttr> src, ArrayRef<DimAttr> dst,
    const SmallVector<int64_t>& srcToDst, const SmallVector<int64_t>& dstToSrc,
    MLIRContext* ctx) {
  // Remaining extent and running stride of each dst tensor dim.
  llvm::DenseMap<int64_t, int64_t> remaining, stride;
  for (DimAttr d : dst) {
    if (!isTraversal(d)) continue;
    remaining[d.getDim()] *= 1;  // touch
    remaining[d.getDim()] = remaining[d.getDim()] == 0
                                ? d.getSize()
                                : remaining[d.getDim()] * d.getSize();
    stride[d.getDim()] = remaining[d.getDim()];
  }
  // The dst dims that a src replication piece may become, used in order.
  SmallVector<int64_t> replTargets = broadcastDims(dstToSrc);

  SmallVector<DimAttr> out;
  for (DimAttr piece : src) {
    if (piece.isGap()) {
      out.push_back(piece);
      continue;
    }
    int64_t target;
    if (isRepl(piece)) {
      // Take the first broadcast target with extent left; if none, the
      // replication stays replication (the other side is replicated too).
      target = -1;
      for (int64_t d : replTargets) {
        // A unit dim can absorb no copies: mirroring onto it would leave the
        // whole replication over as replication anyway, so skip it and keep
        // the piece as it is.
        if (remaining.count(d) && remaining[d] > 1) {
          target = d;
          break;
        }
      }
      if (target < 0) {
        out.push_back(piece);
        continue;
      }
    } else {
      if (piece.getDim() >= static_cast<int64_t>(srcToDst.size())) {
        return std::nullopt;
      }
      target = srcToDst[piece.getDim()];
      if (target == OperatorAlignmentMap::kRepeatedDim) {
        // The map broadcasts this src dim over the dst side: dst holds it as
        // replication of the same extent.
        out.push_back(mk(ctx, -1, piece.getSize(), piece.getStride()));
        continue;
      }
      if (!remaining.count(target) || remaining[target] <= 0) {
        return std::nullopt;
      }
    }
    const int64_t extent = piece.getSize();
    int64_t size, pieceStride;
    if (remaining[target] <= extent) {
      size = remaining[target];
      pieceStride = stride[target] / size;
      remaining[target] = 0;
      stride[target] /= size;
    } else {
      size = extent;
      pieceStride = stride[target] / size;
      remaining[target] /= size;
      stride[target] /= size;
    }
    // Add a replication piece for the leftover extent (extent / size).
    if (isRepl(piece) && extent > size) {
      out.push_back(mk(ctx, -1, extent / size, piece.getStride() * size));
    }
    // A paired traversal piece takes the mirrored side's stride.
    out.push_back(mk(ctx, target, size,
                     isTraversal(piece) ? piece.getStride() : pieceStride));
  }
  return out;
}
}  // namespace

std::optional<AlignedDims> alignedDims(const OperatorAlignmentMap& map,
                                       ArrayRef<DimAttr> lhsDims,
                                       ArrayRef<DimAttr> rhsDims,
                                       MLIRContext* ctx) {
  auto forRhs = mirrorDims(lhsDims, rhsDims, map.lhsToRhs, map.rhsToLhs, ctx);
  auto forLhs = mirrorDims(rhsDims, lhsDims, map.rhsToLhs, map.lhsToRhs, ctx);
  if (!forRhs || !forLhs) return std::nullopt;
  return AlignedDims{std::move(*forRhs), std::move(*forLhs)};
}

namespace {
// Helper function to grow one side to `shapeLen` data elements (the
// reference's replicate_dimensions inner loop). Gaps become replication right
// to left; any missing factor is added as a new outermost replication piece.
SmallVector<DimAttr> fillToShape(ArrayRef<DimAttr> dims, int64_t shapeLen,
                                 MLIRContext* ctx) {
  shapeLen /= fillLen(dims);
  if (shapeLen <= 1) return SmallVector<DimAttr>(dims.begin(), dims.end());
  SmallVector<DimAttr> out;
  int64_t replicated = 1;
  for (size_t p = dims.size(); p-- > 0;) {
    DimAttr d = dims[p];
    if (shapeLen > 1 && d.isGap()) {
      if (d.getSize() <= shapeLen) {
        shapeLen /= d.getSize();
        out.insert(out.begin(), mk(ctx, -1, d.getSize(), replicated));
        replicated *= d.getSize();
      } else {
        // Split the gap: the inner part becomes replication, the outer part
        // stays gap.
        const int64_t inner = shapeLen;
        const int64_t outer = d.getSize() / shapeLen;
        shapeLen = 1;
        out.insert(out.begin(), mk(ctx, -1, inner, replicated));
        out.insert(out.begin(), mk(ctx, -2, outer, 1));
        replicated *= inner;
      }
    } else {
      out.insert(out.begin(), d);
    }
  }
  if (shapeLen > 1) out.insert(out.begin(), mk(ctx, -1, shapeLen, replicated));
  return out;
}
}  // namespace

FailureOr<ReplicatedPair> replicateForAlignment(const OperatorAlignmentMap& map,
                                                LayoutAttr lhs, LayoutAttr rhs,
                                                ArrayRef<int64_t> lhsShape,
                                                ArrayRef<int64_t> rhsShape) {
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return failure();
  MLIRContext* ctx = lhs.getContext();
  SmallVector<DimAttr> a = layoutDims(lhs);
  SmallVector<DimAttr> b = layoutDims(rhs);

  // Pad the smaller side with gap ciphertexts so both cover the same
  // capacity (the reference's match_kernel_dims).
  const int64_t capA = capacity(a), capB = capacity(b);
  if (capA < capB) a.insert(a.begin(), mk(ctx, -2, capB / capA, 1));
  if (capB < capA) b.insert(b.begin(), mk(ctx, -2, capA / capB, 1));

  // The aligned shape: one extent per aligned dim pair.
  int64_t alignedLen = 1;
  for (int64_t d = 0; d < static_cast<int64_t>(map.lhsToRhs.size()); ++d) {
    if (d < static_cast<int64_t>(lhsShape.size())) alignedLen *= lhsShape[d];
  }
  for (int64_t d = 0; d < static_cast<int64_t>(map.rhsToLhs.size()); ++d) {
    if (map.rhsToLhs[d] == OperatorAlignmentMap::kRepeatedDim &&
        d < static_cast<int64_t>(rhsShape.size())) {
      alignedLen *= rhsShape[d];
    }
  }
  const int64_t shapeLen = std::max({fillLen(a), fillLen(b), alignedLen});

  SmallVector<DimAttr> fa = fillToShape(a, shapeLen, ctx);
  SmallVector<DimAttr> fb = fillToShape(b, shapeLen, ctx);
  if (fillLen(fa) != fillLen(fb)) return failure();

  SmallVector<int64_t> lhsRolls, rhsRolls;
  if (auto r = lhs.getRolls())
    lhsRolls.assign(r.asArrayRef().begin(), r.asArrayRef().end());
  if (auto r = rhs.getRolls())
    rhsRolls.assign(r.asArrayRef().begin(), r.asArrayRef().end());
  // Prepended pieces shift piece arguments right.
  auto shiftRolls = [](SmallVector<int64_t>& rolls, int64_t by) {
    for (int64_t& e : rolls) {
      RollArg arg = decodeRollArg(e);
      if (!arg.isAxis) e = encodeRollArg({false, arg.index + by});
    }
  };
  shiftRolls(lhsRolls,
             static_cast<int64_t>(fa.size()) - layoutDims(lhs).size());
  shiftRolls(rhsRolls,
             static_cast<int64_t>(fb.size()) - layoutDims(rhs).size());

  LayoutAttr outA = mergeAdjacentLayoutDims(
      LayoutAttr::getCanonical(ctx, fa, lhs.getN(), lhsRolls));
  LayoutAttr outB = mergeAdjacentLayoutDims(
      LayoutAttr::getCanonical(ctx, fb, rhs.getN(), rhsRolls));
  if (!outA || !outB) return failure();
  return ReplicatedPair{outA, outB};
}

std::optional<LayoutAttr> applySumRoll(LayoutAttr layout, int64_t sumDim) {
  if (!layout) return std::nullopt;
  MLIRContext* ctx = layout.getContext();
  SmallVector<DimAttr> dims = layoutDims(layout);
  const size_t ctLen = inferCtPrefixLen(dims, layout.getN());

  // Preconditions: a replication piece on the ciphertext side, a slot piece
  // of sumDim, and no roll already touching sumDim.
  int64_t ctRepl = -1;
  for (size_t p = 0; p < ctLen; ++p) {
    if (isRepl(dims[p])) {
      if (ctRepl >= 0) return std::nullopt;  // the reference asserts one
      ctRepl = p;
    }
  }
  if (ctRepl < 0) return std::nullopt;
  int64_t slotSum = -1;
  for (size_t p = ctLen; p < dims.size(); ++p) {
    if (dims[p].getDim() == sumDim &&
        (slotSum < 0 || dims[p].getSize() > dims[slotSum].getSize())) {
      slotSum = p;
    }
  }
  if (slotSum < 0) return std::nullopt;
  for (const RollSpec& roll : getRollSpecs(layout)) {
    for (const RollArg& arg : {roll.from, roll.by}) {
      if (arg.isAxis ? arg.index == sumDim
                     : dims[arg.index].getDim() == sumDim) {
        return std::nullopt;
      }
    }
  }

  // Match extents by splitting the larger side; the matched halves swap.
  DimAttr ctPiece = dims[ctRepl];
  DimAttr sumPiece = dims[slotSum];
  SmallVector<DimAttr> out;
  int64_t fromPos, byPos;
  if (sumPiece.getSize() == ctPiece.getSize()) {
    out = dims;
    out[ctRepl] = sumPiece;
    out[slotSum] = ctPiece;
    fromPos = ctRepl;
    byPos = slotSum;
  } else if (sumPiece.getSize() < ctPiece.getSize()) {
    // Split the ct replication: [matched][rest].
    const int64_t e1 = sumPiece.getSize();
    const int64_t e2 = ctPiece.getSize() / e1;
    DimAttr matched = mk(ctx, -1, e1, ctPiece.getStride());
    DimAttr rest = mk(ctx, -1, e2, e1 * ctPiece.getStride());
    out.assign(dims.begin(), dims.begin() + ctRepl);
    out.push_back(sumPiece);  // sum piece takes the matched slot
    out.push_back(rest);
    out.append(dims.begin() + ctRepl + 1, dims.end());
    fromPos = ctRepl;
    byPos = slotSum + 1;  // one piece was inserted before it
    out[byPos] = matched;
  } else {
    // Split the slot sum piece: [hi (matched)][lo].
    const int64_t e1 = ctPiece.getSize();
    const int64_t e2 = sumPiece.getSize() / e1;
    DimAttr hi = mk(ctx, sumDim, e1, e2 * sumPiece.getStride());
    DimAttr lo = mk(ctx, sumDim, e2, sumPiece.getStride());
    out = dims;
    out[ctRepl] = hi;
    out.erase(out.begin() + slotSum);
    out.insert(out.begin() + slotSum, ctPiece);
    out.insert(out.begin() + slotSum + 1, lo);
    fromPos = ctRepl;
    byPos = slotSum;
  }

  SmallVector<int64_t> rolls;
  if (auto r = layout.getRolls())
    rolls.assign(r.asArrayRef().begin(), r.asArrayRef().end());
  // Existing piece arguments at or past an insertion shift right.
  for (int64_t& e : rolls) {
    RollArg arg = decodeRollArg(e);
    if (arg.isAxis) continue;
    int64_t idx = arg.index;
    if (sumPiece.getSize() < ctPiece.getSize() && idx > ctRepl) ++idx;
    if (sumPiece.getSize() > ctPiece.getSize() && idx > slotSum) ++idx;
    e = encodeRollArg({false, idx});
  }
  rolls.push_back(encodeRollArg({false, fromPos}));
  rolls.push_back(encodeRollArg({false, byPos}));
  LayoutAttr rolled = LayoutAttr::getCanonical(ctx, out, layout.getN(), rolls);
  if (!rolled) return std::nullopt;
  return rolled;
}

namespace {
SmallVector<int64_t> rollVec(LayoutAttr layout) {
  SmallVector<int64_t> out;
  if (auto r = layout.getRolls()) {
    out.assign(r.asArrayRef().begin(), r.asArrayRef().end());
  }
  return out;
}
bool hasRolls(LayoutAttr layout) {
  return layout.getRolls() && !layout.getRolls().empty();
}

// Applies rolls to `layout` so its pieces reach `target`'s order, which is a
// reorder of the same pieces. Each out-of-place traversal piece is rolled by
// the piece at its target position. Existing rolls are updated as in the
// reference's roll_update. Nullopt when the pieces differ or an out-of-place
// piece meets a partner with a different extent.
std::optional<LayoutAttr> applyRoll(LayoutAttr layout,
                                    ArrayRef<DimAttr> target) {
  MLIRContext* ctx = layout.getContext();
  SmallVector<DimAttr> dims = layoutDims(layout);
  // Match on the pieces and swap by their full-list positions.
  SmallVector<size_t> fullPos;  // filtered index -> full index
  SmallVector<DimAttr> pieces, targetPieces;
  for (size_t p = 0; p < dims.size(); ++p) {
    fullPos.push_back(p);
    pieces.push_back(dims[p]);
  }
  for (DimAttr d : target) {
    if (d.getSize() != 1) targetPieces.push_back(d);
  }
  if (pieces.size() != targetPieces.size()) return std::nullopt;
  for (DimAttr d : pieces) {
    if (!llvm::is_contained(targetPieces, d)) return std::nullopt;
  }
  for (DimAttr d : targetPieces) {
    if (!llvm::is_contained(pieces, d)) return std::nullopt;
  }
  SmallVector<int64_t> rolls = rollVec(layout);

  for (size_t ti = 0; ti < targetPieces.size(); ++ti) {
    DimAttr piece = targetPieces[ti];
    if (!isTraversal(piece)) continue;
    auto it = llvm::find(pieces, piece);
    const size_t curIdx = it - pieces.begin();
    if (curIdx == ti) continue;
    const size_t cur = fullPos[curIdx];
    const size_t i = fullPos[ti];
    DimAttr partner = dims[i];
    if (partner.getSize() != piece.getSize()) return std::nullopt;

    // Rebase existing rolls across the swap of positions cur <-> i.
    SmallVector<int64_t> rebased;
    for (size_t r = 0; r + 1 < rolls.size(); r += 2) {
      RollArg from = decodeRollArg(rolls[r]);
      RollArg by = decodeRollArg(rolls[r + 1]);
      RollArg newFrom = from;
      if (!from.isAxis) {  // follows its piece
        if (from.index == static_cast<int64_t>(cur))
          newFrom.index = i;
        else if (from.index == static_cast<int64_t>(i))
          newFrom.index = cur;
      }
      RollArg newBy = by;  // positional
      if (!newFrom.isAxis && !newBy.isAxis && newFrom.index == newBy.index) {
        if (by.index == static_cast<int64_t>(cur))
          newBy.index = i;
        else if (by.index == static_cast<int64_t>(i))
          newBy.index = cur;
      }
      rebased.push_back(encodeRollArg(newFrom));
      rebased.push_back(encodeRollArg(newBy));
    }
    rolls = std::move(rebased);
    std::swap(dims[cur], dims[i]);
    std::swap(pieces[curIdx], pieces[ti]);
    // Swapping the roll: the piece (now at i) by the partner (now at cur).
    const int64_t newFrom = encodeRollArg({false, static_cast<int64_t>(i)});
    const int64_t newBy = encodeRollArg({false, static_cast<int64_t>(cur)});
    bool present = false;
    for (size_t r = 0; r + 1 < rolls.size(); r += 2) {
      if (rolls[r] == newFrom && rolls[r + 1] == newBy) present = true;
    }
    if (!present) {
      rolls.push_back(newFrom);
      rolls.push_back(newBy);
    }
  }
  LayoutAttr out = LayoutAttr::getCanonical(ctx, dims, layout.getN(), rolls);
  if (!out) return std::nullopt;
  return out;
}
}  // namespace

SmallVector<AlignedPair, 2> rollToAlign(const OperatorAlignmentMap& map,
                                        LayoutAttr lhs, LayoutAttr rhs) {
  SmallVector<AlignedPair, 2> out;
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return out;
  // Early exit if a swap roll cannot fix differing rolls or differing strides.
  LayoutAttr lhsMerged = mergeAdjacentLayoutDims(lhs);
  LayoutAttr rhsMerged = mergeAdjacentLayoutDims(rhs);
  ArrayAttr lhsMergedDims = lhsMerged.getDims();
  ArrayAttr rhsMergedDims = rhsMerged.getDims();
  if (lhsMergedDims.size() == rhsMergedDims.size() &&
      !checkRollAlignment(map, lhsMerged, rhsMerged, lhsMergedDims,
                          rhsMergedDims)) {
    return out;
  }
  auto aligned =
      alignedDims(map, layoutDims(lhs), layoutDims(rhs), lhs.getContext());
  if (!aligned) return out;
  {
    SmallVector<DimAttr> a = layoutDims(lhs);
    for (size_t p = 0; p < a.size() && p < aligned->forLhs.size(); ++p) {
      DimAttr x = a[p], y = aligned->forLhs[p];
      if (isTraversal(x) && isTraversal(y) && x.getStride() != y.getStride()) {
        return out;
      }
    }
  }
  if (auto rolledRhs = applyRoll(rhs, aligned->forRhs)) {
    if (isOperatorAligned(map, lhs, *rolledRhs))
      out.push_back({lhs, *rolledRhs});
  }
  if (auto rolledLhs = applyRoll(lhs, aligned->forLhs)) {
    if (isOperatorAligned(map, *rolledLhs, rhs))
      out.push_back({*rolledLhs, rhs});
  }
  return out;
}

// Restates `dims` so its pieces correspond one to one with `reference`
// Unit pieces on either side are carried through. Returns nullopt when the
// lists cannot be made to correspond, or when `dims` carries rolls.
static std::optional<SmallVector<DimAttr>> restateToMatch(
    ArrayRef<DimAttr> reference, LayoutAttr layout, MLIRContext* ctx) {
  if (layout.getRolls() && !layout.getRolls().empty()) return std::nullopt;
  SmallVector<DimAttr> dims = layoutDims(layout);
  size_t i = 0, j = 0;
  while (i < reference.size() && j < dims.size()) {
    const int64_t want = reference[i].getSize(), have = dims[j].getSize();
    if (want == have) {
      ++i;
      ++j;
      continue;
    }
    if (have < want || have % want != 0) return std::nullopt;
    DimAttr piece = dims[j];
    if (!piece.isReplicate() && !piece.isGap()) return std::nullopt;
    // The piece facing the reference comes first, with the remainder inside
    // it: [R:64:1] facing [R:4:16] becomes [R:4:16],[R:16:1], the outer
    // part's stride being the inner part's extent, as the mirror writes it.
    const int64_t rest = have / want;
    DimAttr outer =
        DimAttr::get(ctx, piece.getDim(), want, piece.getStride() * rest);
    DimAttr inner = DimAttr::get(ctx, piece.getDim(), rest, piece.getStride());
    dims[j] = outer;
    dims.insert(dims.begin() + j + 1, inner);
    ++i;
    ++j;
  }
  return dims;
}

std::optional<LayoutAttr> matchPublicLayout(const OperatorAlignmentMap& map,
                                            LayoutAttr lhs, LayoutAttr rhs,
                                            bool matchLhs) {
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return std::nullopt;
  MLIRContext* ctx = lhs.getContext();
  std::optional<AlignedDims> aligned =
      alignedDims(map, layoutDims(lhs), layoutDims(rhs), ctx);
  if (!aligned) return std::nullopt;
  ArrayRef<DimAttr> dims = matchLhs ? aligned->forLhs : aligned->forRhs;
  LayoutAttr other = matchLhs ? rhs : lhs;
  SmallVector<int64_t> rolls;
  if (auto r = other.getRolls()) {
    for (int64_t encoded : r.asArrayRef()) {
      RollArg arg = decodeRollArg(encoded);
      if (arg.isAxis) return std::nullopt;
      if (arg.index < 0 || arg.index >= static_cast<int64_t>(dims.size())) {
        return std::nullopt;
      }
      rolls.push_back(encoded);
    }
  }
  LayoutAttr out = LayoutAttr::getCanonical(ctx, dims, lhs.getN(), rolls);
  if (!out) return std::nullopt;
  return out;
}

SmallVector<AlignedPair> alignPair(const OperatorAlignmentMap& map,
                                   LayoutAttr lhs, LayoutAttr rhs) {
  SmallVector<AlignedPair> out;
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return out;
  if (isOperatorAligned(map, lhs, rhs)) {
    MLIRContext* ctx = lhs.getContext();
    if (layoutDims(lhs).size() != layoutDims(rhs).size()) {
      if (auto r = restateToMatch(layoutDims(lhs), rhs, ctx)) {
        SmallVector<Attribute> attrs(r->begin(), r->end());
        if (LayoutAttr rr = LayoutAttr::get(ctx, ArrayAttr::get(ctx, attrs),
                                            rhs.getN(), rhs.getRolls())) {
          out.push_back({lhs, rr});
          return out;
        }
      }
      if (auto l = restateToMatch(layoutDims(rhs), lhs, ctx)) {
        SmallVector<Attribute> attrs(l->begin(), l->end());
        if (LayoutAttr ll = LayoutAttr::get(ctx, ArrayAttr::get(ctx, attrs),
                                            lhs.getN(), lhs.getRolls())) {
          out.push_back({ll, rhs});
          return out;
        }
      }
    }
    out.push_back({lhs, rhs});
    return out;
  }
  MLIRContext* ctx = lhs.getContext();
  // Conversion candidates are conversions by only moving pieces.
  if (!hasRolls(lhs) && !hasRolls(rhs)) {
    if (auto aligned =
            alignedDims(map, layoutDims(lhs), layoutDims(rhs), ctx)) {
      LayoutAttr convRhs = mergeAdjacentLayoutDims(
          LayoutAttr::getCanonical(ctx, aligned->forRhs, rhs.getN()));
      LayoutAttr convLhs = mergeAdjacentLayoutDims(
          LayoutAttr::getCanonical(ctx, aligned->forLhs, lhs.getN()));
      if (convRhs && isOperatorAligned(map, lhs, convRhs)) {
        out.push_back({lhs, convRhs});
      }
      if (convLhs && isOperatorAligned(map, convLhs, rhs)) {
        out.push_back({convLhs, rhs});
      }
    }
  }
  // Repack candidates are conversions that are moved to public inputs.
  auto withPartner = [&](LayoutAttr repacked, LayoutAttr partner,
                         bool repackedIsLhs) {
    std::optional<SmallVector<DimAttr>> restated =
        restateToMatch(layoutDims(repacked), partner, ctx);
    LayoutAttr partnerOut = partner;
    if (restated) {
      SmallVector<Attribute> attrs(restated->begin(), restated->end());
      partnerOut = LayoutAttr::get(ctx, ArrayAttr::get(ctx, attrs),
                                   partner.getN(), partner.getRolls());
      if (!partnerOut) partnerOut = partner;
    }
    LayoutAttr l = repackedIsLhs ? repacked : partnerOut;
    LayoutAttr r = repackedIsLhs ? partnerOut : repacked;
    if (isOperatorAligned(map, l, r)) out.push_back({l, r});
  };
  if (std::optional<LayoutAttr> matched =
          matchPublicLayout(map, lhs, rhs, /*matchLhs=*/true)) {
    withPartner(*matched, rhs, /*repackedIsLhs=*/true);
  }
  if (std::optional<LayoutAttr> matched =
          matchPublicLayout(map, lhs, rhs, /*matchLhs=*/false)) {
    withPartner(*matched, lhs, /*repackedIsLhs=*/false);
  }
  // Roll candidates are conversions that require rolls.
  for (const AlignedPair& p : rollToAlign(map, lhs, rhs)) out.push_back(p);
  SmallVector<AlignedPair> unique;
  for (const AlignedPair& p : out) {
    bool seen = false;
    for (const AlignedPair& q : unique) {
      seen |= q.lhs == p.lhs && q.rhs == p.rhs;
    }
    if (!seen) unique.push_back(p);
  }
  return unique;
}

std::optional<LayoutAttr> outputLayout(const OperatorAlignmentMap& map,
                                       bool isMatmul, LayoutAttr lhs,
                                       LayoutAttr rhs, int64_t lhsSumDim,
                                       int64_t rhsSumDim) {
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return std::nullopt;
  MLIRContext* ctx = lhs.getContext();
  SmallVector<DimAttr> a = layoutDims(lhs);
  SmallVector<DimAttr> b = layoutDims(rhs);

  // A summation in the ciphertext dimensions removes the piece.
  // A summation in the slot dimensions results in a gap piece.
  const size_t ctLen = inferCtPrefixLen(a, lhs.getN());

  SmallVector<DimAttr> outDims;
  SmallVector<int64_t> posMap(a.size(), -1), posMapB(b.size(), -1);
  // The pair arrives aligned, so the two lists correspond one to one and the
  // walk steps them together.
  size_t ia = 0, ib = 0;
  while (ia < a.size() || ib < b.size()) {
    if (ia >= a.size() || ib >= b.size()) return std::nullopt;
    const size_t i = ia, j = ib;
    DimAttr x = a[ia++];
    DimAttr y = b[ib++];
    if (isMatmul && i < ctLen &&
        ((isTraversal(x) && x.getDim() == lhsSumDim) ||
         (isTraversal(y) && y.getDim() == rhsSumDim))) {
      continue;
    }
    posMap[i] = static_cast<int64_t>(outDims.size());
    posMapB[j] = static_cast<int64_t>(outDims.size());
    if (isMatmul) {
      if (isTraversal(x) && x.getDim() == lhsSumDim) {
        outDims.push_back(mk(ctx, -2, x.getSize(), x.getStride()));
      } else if (isTraversal(y) && y.getDim() == rhsSumDim) {
        outDims.push_back(mk(ctx, -2, y.getSize(), y.getStride()));
      } else if (isTraversal(x) && !isTraversal(y)) {
        outDims.push_back(x);
      } else if (!isTraversal(x) && isTraversal(y)) {
        outDims.push_back(y);
      } else if (x.isGap() || y.isGap()) {
        outDims.push_back(mk(ctx, -2, x.getSize(), x.getStride()));
      } else {
        // Both replication: the output holds neither dim here.
        outDims.push_back(mk(ctx, -2, x.getSize(), x.getStride()));
      }
    } else {
      if (isTraversal(x) || (!isTraversal(y))) {
        outDims.push_back(x);
      } else {
        outDims.push_back(y);  // lhs replication over an rhs traversal dim
      }
    }
  }

  // Rolls along the summation dimension are dropped.
  SmallVector<int64_t> rolls;
  auto fromIsSum = [&](ArrayRef<DimAttr> dims, const RollArg& from,
                       int64_t sumDim) {
    return from.isAxis ? from.index == sumDim
                       : dims[from.index].getDim() == sumDim;
  };
  auto remap = [&](RollArg arg, ArrayRef<int64_t> map, bool& ok) {
    if (arg.isAxis) return arg;
    if (arg.index < 0 || arg.index >= static_cast<int64_t>(map.size()) ||
        map[arg.index] < 0) {
      ok = false;
      return arg;
    }
    arg.index = map[arg.index];
    return arg;
  };
  auto append = [&](LayoutAttr layout, ArrayRef<DimAttr> dims, int64_t sumDim,
                    ArrayRef<int64_t> map) {
    for (const RollSpec& r : getRollSpecs(layout)) {
      if (isMatmul && fromIsSum(dims, r.from, sumDim)) continue;
      bool ok = true;
      const RollArg fromArg = remap(r.from, map, ok);
      const RollArg byArg = remap(r.by, map, ok);
      if (!ok) continue;
      const int64_t f = encodeRollArg(fromArg), by = encodeRollArg(byArg);
      bool dup = false;
      for (size_t k = 0; k + 1 < rolls.size(); k += 2) {
        if (rolls[k] == f && rolls[k + 1] == by) dup = true;
      }
      if (!dup) {
        rolls.push_back(f);
        rolls.push_back(by);
      }
    }
  };
  append(lhs, a, lhsSumDim, posMap);
  if (isMatmul) append(rhs, b, rhsSumDim, posMapB);

  LayoutAttr out = LayoutAttr::getCanonical(ctx, outDims, lhs.getN(), rolls);
  if (!out) return std::nullopt;
  return mergeAdjacentLayoutDims(out);
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
