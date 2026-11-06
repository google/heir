#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/RotationGroupKernel.h"
#include "lib/Dialect/TensorExt/Transforms/ShiftScheme.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/Graph/Graph.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"       // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"       // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

#define DEBUG_TYPE "implement-shift-network"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_IMPLEMENTSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

ShiftScheme VosVosErkinShiftNetworks::findShiftScheme(
    const Mapping& mapping, ArrayRef<int64_t> shiftOrder) {
  CacheKey cacheKey = makeCacheKey(mapping, shiftOrder);
  if (schemeCache.count(cacheKey)) {
    return schemeCache[cacheKey];
  }

  ShiftStrategy strategy = evaluateShiftStrategy(mapping, shiftOrder);

  // Create a graph whose vertices are the input indices to remap, and
  // whose edges are conflicts: an edge being present means the two indices
  // cannot participate in the same rotation group.
  graph::UndirectedGraph<CtSlot> conflictGraph;
  for (const auto& [target, source] : mapping.getTargetToSource()) {
    conflictGraph.addVertex(source);
  }
  for (const auto& [roundNum, round] : llvm::enumerate(strategy.getRounds())) {
    if (roundNum == 0) continue;

    // Sources the round moves to the same position conflict.
    std::map<CtSlot, SmallVector<CtSlot>> sourcesAt;
    for (const auto& [sourceShift, position] : round.positions)
      sourcesAt[position].push_back(sourceShift.source);
    for (const auto& [position, sources] : sourcesAt) {
      for (auto [i, source] : llvm::enumerate(sources)) {
        for (CtSlot other : ArrayRef(sources).drop_front(i + 1)) {
          if (source == other) continue;
          LLVM_DEBUG(llvm::dbgs()
                     << "Round " << roundNum << ": collision between {"
                     << source.ct << "," << source.slot << "} and {" << other.ct
                     << "," << other.slot << "} at {" << position.ct << ","
                     << position.slot << "}\n");
          conflictGraph.addEdge(source, other);
        }
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Conflict graph:\n";
    for (CtSlot vertex : conflictGraph.getVertices()) {
      llvm::dbgs() << vertex.ct << "," << vertex.slot << " <-> {";
      for (CtSlot neighbor : conflictGraph.edgesIncidentTo(vertex)) {
        llvm::dbgs() << neighbor.ct << "," << neighbor.slot << "; ";
      }
      llvm::dbgs() << "}\n";
    }
  });

  graph::GreedyGraphColoring<CtSlot> colorer;
  std::unordered_map<CtSlot, int> coloring = colorer.color(conflictGraph);

  SmallVector<RotationGroup> resultRotationGroups;
  resultRotationGroups.reserve(5);
  for (const auto& entry : coloring) {
    CtSlot source = entry.first;
    int64_t color = entry.second;
    if (color >= resultRotationGroups.size()) {
      resultRotationGroups.resize(color + 1);
    }
    resultRotationGroups[color].insert(source);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Splitting mapping into rotation groups:\n";
    for (int i = 0; i < resultRotationGroups.size(); i++) {
      llvm::dbgs() << "Group " << i << ": ";
      llvm::SmallVector<CtSlot> group = llvm::SmallVector<CtSlot>(
          resultRotationGroups[i].begin(), resultRotationGroups[i].end());
      llvm::sort(group);
      for (CtSlot source : group) {
        llvm::dbgs() << "(" << source.ct << "," << source.slot << ") ";
      }
      llvm::dbgs() << "\n";
    }
  });

  ShiftScheme scheme{resultRotationGroups, strategy};
  schemeCache[cacheKey] = scheme;
  return schemeCache[cacheKey];
}

ShiftScheme VosVosErkinShiftNetworks::findBestShiftScheme(
    const Mapping& mapping, std::size_t randomSeed, unsigned randomTries) {
  SmallVector<int64_t> initShiftOrder = defaultShiftOrder(
      mapping.getCiphertextSize() * mapping.getNumCiphertexts());

  std::size_t numRoundsMin = std::numeric_limits<std::size_t>::max();
  SmallVector<int64_t> bestShiftOrder;

  std::seed_seq seq{randomSeed};
  std::mt19937 g(seq);

  for (unsigned i = 0; i < randomTries; ++i) {
    // In order to get a uniform distribution over all permutations using a
    // Fisher-Yates shuffle we have to apply it to the original vector in each
    // iteration.
    SmallVector<int64_t> shiftOrder = initShiftOrder;

    std::shuffle(shiftOrder.begin(), shiftOrder.end(), g);

    ShiftStrategy strategy = evaluateShiftStrategy(mapping, shiftOrder);

    std::size_t numRounds = strategy.getRounds().size();
    if (numRounds < numRoundsMin) {
      numRoundsMin = numRounds;
      bestShiftOrder = shiftOrder;
    }
  }

  return findShiftScheme(mapping, bestShiftOrder);
}

ShiftStrategy VosVosErkinShiftNetworks::evaluateShiftStrategy(
    const Mapping& mapping, ArrayRef<int64_t> shiftOrder) {
  CacheKey cacheKey = makeCacheKey(mapping, shiftOrder);
  if (strategyCache.count(cacheKey)) {
    return strategyCache[cacheKey];
  }

  ShiftStrategy strategy(mapping.getCiphertextSize(),
                         mapping.getNumCiphertexts(), shiftOrder);
  strategy.evaluate(mapping);
  strategyCache[cacheKey] = strategy;
  return strategy;
}

VosVosErkinShiftNetworks::CacheKey VosVosErkinShiftNetworks::makeCacheKey(
    const Mapping& mapping, ArrayRef<int64_t> shiftOrder) {
  FrozenVector<int64_t> frozenShiftOrder(shiftOrder);
  return std::make_pair(mapping, frozenShiftOrder);
}

bool tryPopulateFactoredMapping(const presburger::IntegerRelation& rel,
                                Mapping& mapping) {
  if (rel.getNumDomainVars() != 2 || rel.getNumRangeVars() != 2 ||
      rel.getNumSymbolVars() != 0 || rel.getNumLocalVars() == 0) {
    return false;
  }

  unsigned localOffset = rel.getVarKindOffset(presburger::VarKind::Local);
  unsigned numLocals = rel.getNumLocalVars();
  unsigned numCols = rel.getNumCols();
  unsigned constCol = numCols - 1;

  // Identify shared intermediate local variables: those that appear directly in
  // at least one constraint touching the source (domain, cols 0..1) and at
  // least one constraint touching the target (range, cols 2..3).
  SmallVector<unsigned> localTouchMask(numLocals, 0);
  for (int isEq = 0; isEq < 2; ++isEq) {
    unsigned numRows = isEq ? rel.getNumEqualities() : rel.getNumInequalities();
    for (unsigned r = 0; r < numRows; ++r) {
      auto at = [&](unsigned c) {
        return isEq ? rel.atEq64(r, c) : rel.atIneq64(r, c);
      };
      unsigned endpointMask = 0;
      if (at(0) != 0 || at(1) != 0) endpointMask |= 1u;
      if (at(2) != 0 || at(3) != 0) endpointMask |= 2u;
      if (endpointMask == 0) continue;
      for (unsigned k = 0; k < numLocals; ++k) {
        if (at(localOffset + k) != 0) localTouchMask[k] |= endpointMask;
      }
    }
  }

  SmallVector<unsigned> sharedLocals;
  for (unsigned k = 0; k < numLocals; ++k) {
    if (localTouchMask[k] == 3u) sharedLocals.push_back(k);
  }
  if (sharedLocals.empty()) return false;

  // side[c]: 0 = shared, 1 = src, 2 = dst, -1 = unassigned
  SmallVector<int> side(constCol, -1);
  side[0] = 1;
  side[1] = 1;
  side[2] = 2;
  side[3] = 2;
  for (unsigned k : sharedLocals) side[localOffset + k] = 0;

  auto rowTouchMask = [&](bool isEq, unsigned r) -> unsigned {
    unsigned mask = 0;
    for (unsigned c = 0; c < constCol; ++c) {
      int64_t v = isEq ? rel.atEq64(r, c) : rel.atIneq64(r, c);
      if (v != 0 && side[c] > 0) mask |= (1u << side[c]);
    }
    return mask;
  };

  bool progress = true;
  while (progress) {
    progress = false;
    for (int isEq = 0; isEq < 2; ++isEq) {
      unsigned numRows =
          isEq ? rel.getNumEqualities() : rel.getNumInequalities();
      for (unsigned r = 0; r < numRows; ++r) {
        unsigned mask = rowTouchMask(isEq, r);
        if (mask != 2u && mask != 4u) continue;
        int rowSide = (mask == 2u) ? 1 : 2;
        for (unsigned c = localOffset; c < constCol; ++c) {
          int64_t v = isEq ? rel.atEq64(r, c) : rel.atIneq64(r, c);
          if (v != 0 && side[c] == -1) {
            side[c] = rowSide;
            progress = true;
          }
        }
      }
    }
  }

  SmallVector<unsigned> srcLocals, dstLocals;
  for (unsigned k = 0; k < numLocals; ++k) {
    int s = side[localOffset + k];
    if (s == -1) return false;
    if (s == 1) srcLocals.push_back(k);
    if (s == 2) dstLocals.push_back(k);
  }

  for (int isEq = 0; isEq < 2; ++isEq) {
    unsigned numRows = isEq ? rel.getNumEqualities() : rel.getNumInequalities();
    for (unsigned r = 0; r < numRows; ++r) {
      if ((rowTouchMask(isEq, r) & 6u) == 6u) return false;
    }
  }

  unsigned numShared = sharedLocals.size();
  SmallVector<int> srcColMap(constCol, -1), dstColMap(constCol, -1);
  for (unsigned i = 0; i < numShared; ++i) {
    srcColMap[localOffset + sharedLocals[i]] = i;
    dstColMap[localOffset + sharedLocals[i]] = i;
  }
  srcColMap[0] = numShared;
  srcColMap[1] = numShared + 1;
  dstColMap[2] = numShared;
  dstColMap[3] = numShared + 1;
  for (unsigned i = 0; i < srcLocals.size(); ++i) {
    srcColMap[localOffset + srcLocals[i]] = numShared + 2 + i;
  }
  for (unsigned i = 0; i < dstLocals.size(); ++i) {
    dstColMap[localOffset + dstLocals[i]] = numShared + 2 + i;
  }

  presburger::IntegerRelation srcRel(
      presburger::PresburgerSpace::getRelationSpace(numShared, 2, 0,
                                                    srcLocals.size()));
  presburger::IntegerRelation dstRel(
      presburger::PresburgerSpace::getRelationSpace(numShared, 2, 0,
                                                    dstLocals.size()));

  for (int isEq = 0; isEq < 2; ++isEq) {
    unsigned numRows = isEq ? rel.getNumEqualities() : rel.getNumInequalities();
    for (unsigned r = 0; r < numRows; ++r) {
      unsigned mask = rowTouchMask(isEq, r);
      int64_t cst = isEq ? rel.atEq64(r, constCol) : rel.atIneq64(r, constCol);
      if ((mask & 4u) == 0) {
        SmallVector<int64_t> row(srcRel.getNumCols(), 0);
        for (unsigned c = 0; c < constCol; ++c) {
          int64_t v = isEq ? rel.atEq64(r, c) : rel.atIneq64(r, c);
          if (v != 0) row[srcColMap[c]] = v;
        }
        row.back() = cst;
        if (isEq)
          srcRel.addEquality(row);
        else
          srcRel.addInequality(row);
      }
      if ((mask & 2u) == 0) {
        SmallVector<int64_t> row(dstRel.getNumCols(), 0);
        for (unsigned c = 0; c < constCol; ++c) {
          int64_t v = isEq ? rel.atEq64(r, c) : rel.atIneq64(r, c);
          if (v != 0) row[dstColMap[c]] = v;
        }
        row.back() = cst;
        if (isEq)
          dstRel.addEquality(row);
        else
          dstRel.addInequality(row);
      }
    }
  }

  for (unsigned i = 0; i < numShared; ++i) {
    auto srcLb = srcRel.getConstantBound64(presburger::BoundType::LB, i);
    auto srcUb = srcRel.getConstantBound64(presburger::BoundType::UB, i);
    auto dstLb = dstRel.getConstantBound64(presburger::BoundType::LB, i);
    auto dstUb = dstRel.getConstantBound64(presburger::BoundType::UB, i);
    if ((!srcLb && !dstLb) || (!srcUb && !dstUb)) return false;
    int64_t lb =
        (srcLb && dstLb) ? std::max(*srcLb, *dstLb) : (srcLb ? *srcLb : *dstLb);
    int64_t ub =
        (srcUb && dstUb) ? std::min(*srcUb, *dstUb) : (srcUb ? *srcUb : *dstUb);
    if (lb > ub) return false;
    if (!srcLb || *srcLb < lb) {
      SmallVector<int64_t> row(srcRel.getNumCols(), 0);
      row[i] = 1;
      row.back() = -lb;
      srcRel.addInequality(row);
    }
    if (!srcUb || *srcUb > ub) {
      SmallVector<int64_t> row(srcRel.getNumCols(), 0);
      row[i] = -1;
      row.back() = ub;
      srcRel.addInequality(row);
    }
    if (!dstLb || *dstLb < lb) {
      SmallVector<int64_t> row(dstRel.getNumCols(), 0);
      row[i] = 1;
      row.back() = -lb;
      dstRel.addInequality(row);
    }
    if (!dstUb || *dstUb > ub) {
      SmallVector<int64_t> row(dstRel.getNumCols(), 0);
      row[i] = -1;
      row.back() = ub;
      dstRel.addInequality(row);
    }
  }

  PointPairCollector srcCollector(numShared, 2);
  PointPairCollector dstCollector(numShared, 2);
  enumeratePoints(srcRel, srcCollector);
  enumeratePoints(dstRel, dstCollector);

  llvm::sort(srcCollector.points);
  llvm::sort(dstCollector.points);

  size_t sIdx = 0;
  size_t dIdx = 0;
  const auto& srcPts = srcCollector.points;
  const auto& dstPts = dstCollector.points;
  while (sIdx < srcPts.size() && dIdx < dstPts.size()) {
    if (srcPts[sIdx].first < dstPts[dIdx].first) {
      ++sIdx;
      continue;
    }
    if (dstPts[dIdx].first < srcPts[sIdx].first) {
      ++dIdx;
      continue;
    }
    const auto& coord = srcPts[sIdx].first;
    size_t sEnd = sIdx + 1;
    while (sEnd < srcPts.size() && srcPts[sEnd].first == coord) ++sEnd;
    size_t dEnd = dIdx + 1;
    while (dEnd < dstPts.size() && dstPts[dEnd].first == coord) ++dEnd;

    for (size_t di = dIdx; di < dEnd; ++di) {
      CtSlot target{dstPts[di].second[0], dstPts[di].second[1]};
      bool hasIdentity = false;
      for (size_t si = sIdx; si < sEnd; ++si) {
        if (srcPts[si].second[0] == target.ct &&
            srcPts[si].second[1] == target.slot) {
          hasIdentity = true;
          break;
        }
      }
      if (hasIdentity) {
        mapping.add(target, target);
      } else {
        for (size_t si = sIdx; si < sEnd; ++si) {
          CtSlot source{srcPts[si].second[0], srcPts[si].second[1]};
          mapping.add(source, target);
        }
      }
    }
    sIdx = sEnd;
    dIdx = dEnd;
  }
  return true;
}

void populateMappingFromLayoutAttr(const LayoutAttr& layoutAttr,
                                   Mapping& mapping) {
  const presburger::IntegerRelation& rel = layoutAttr.getIntegerRelation();
  if (tryPopulateFactoredMapping(rel, mapping)) return;

  PointPairCollector collector(2, 2);
  enumeratePoints(rel, collector);

  // Put the data from collector into Mapping. Probably can be more efficient
  // here by avoiding a copy and making a custom PointPairCollector that
  // directly adds to mapping.
  for (const auto& [source, target] : collector.points) {
    CtSlot sourceSlot{source[0], source[1]};
    CtSlot targetSlot{target[0], target[1]};
    mapping.add(sourceSlot, targetSlot);
  }
}

void populateMappingFromDenseElementsAttr(
    const DenseIntElementsAttr& denseElementsAttr, Mapping& mapping) {
  // iterate in batches of 4 to get the (ct_source, slot_source,
  // ct_target, slot_target) tuples
  for (auto it = denseElementsAttr.value_begin<int64_t>(),
            e = denseElementsAttr.value_end<int64_t>();
       it != e;) {
    int64_t ctSource = *it++;
    int64_t slotSource = *it++;
    int64_t ctTarget = *it++;
    int64_t slotTarget = *it++;
    mapping.add(CtSlot{ctSource, slotSource}, CtSlot{ctTarget, slotTarget});
  }
}

using MappingCacheKey = std::tuple<Attribute, int64_t, int64_t>;

LogicalResult convertRemapOp(
    RemapOp op, VosVosErkinShiftNetworks& shiftNetworks,
    llvm::DenseMap<MappingCacheKey, Mapping>& mappingCache) {
  LLVM_DEBUG(llvm::dbgs() << "Converting layout op: " << op << "\n");
  ImplicitLocOpBuilder b(op.getLoc(), op.getContext());
  RankedTensorType tensorTy = op.getInput().getType();
  // Since this pass may only run after convert-to-ciphertext-semantics, the
  // input must be a 2D tensor of (ct, slot) shape.
  int64_t numCiphertexts = tensorTy.getDimSize(0);
  int64_t minSlotCount = tensorTy.getDimSize(1);
  auto singleCiphertextType =
      RankedTensorType::get({1, minSlotCount}, tensorTy.getElementType());

  // Extract element type and convert to DagType
  kernel::DagType dagElemType;
  if (auto floatTy = dyn_cast<FloatType>(tensorTy.getElementType())) {
    dagElemType = kernel::DagType::floatTy(floatTy.getWidth());
  } else if (auto intTy = dyn_cast<IntegerType>(tensorTy.getElementType())) {
    dagElemType = kernel::DagType::integer(intTy.getWidth());
  } else {
    return op.emitOpError()
           << "expected tensor with float or integer element type";
  }

  // Populate the mapping with (source, target) pairs
  // This require enumerating over the relation for the op
  MappingCacheKey mappingKey{op.getPermutation(), minSlotCount, numCiphertexts};
  auto mappingIt = mappingCache.find(mappingKey);
  if (mappingIt == mappingCache.end()) {
    Mapping newMapping(minSlotCount, numCiphertexts);
    if (auto layoutAttr = dyn_cast<LayoutAttr>(op.getPermutation())) {
      populateMappingFromLayoutAttr(layoutAttr, newMapping);
    } else if (auto denseElementsAttr =
                   dyn_cast<DenseIntElementsAttr>(op.getPermutation())) {
      populateMappingFromDenseElementsAttr(denseElementsAttr, newMapping);
    } else {
      return op.emitOpError()
             << "requires permutation attribute to be either LayoutAttr or "
                "DenseIntElementsAttr";
    }
    mappingIt =
        mappingCache.try_emplace(mappingKey, std::move(newMapping)).first;
  }
  const Mapping& mapping = mappingIt->second;

  b.setInsertionPointAfter(op);

  // Could add a special case here if the numCiphertexts == 1, using
  // collapse_shape and expand_shape instead of insert/extract slice. No sure
  // if that would be more efficient in the end...

  // Decompose the tensor of ciphertexts into individual values. This is done
  // by extracting the slice of a tensor<kxN> corresponding to one row.
  //
  // Also needed to be compatible with the ArithmeticDag interface and to
  // avoid extract slice ops in the middle of the dag.
  SmallVector<kernel::SSAValue> ciphertexts;
  for (int64_t i = 0; i < numCiphertexts; i++) {
    auto one = b.getIndexAttr(1);
    SmallVector<OpFoldResult> offsets = {b.getIndexAttr(i), b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {one,
                                       b.getIndexAttr(tensorTy.getDimSize(1))};
    SmallVector<OpFoldResult> strides = {one, one};
    auto slice =
        tensor::ExtractSliceOp::create(b, op.getLoc(), singleCiphertextType,
                                       op.getInput(), offsets, sizes, strides);
    ciphertexts.push_back(kernel::SSAValue(slice.getResult()));
  }

  using NodeTy = kernel::ArithmeticDagNode<kernel::SSAValue>;
  SmallVector<std::shared_ptr<NodeTy>> resultNodes;

  const auto targetToSource = mapping.getTargetToSource();

  // Count the rotations the direct path would emit.
  llvm::DenseSet<std::pair<int64_t, int64_t>> uniqueShifts;
  for (const auto& [target, source] : targetToSource) {
    int64_t r = (source.slot - target.slot) % minSlotCount;
    if (r < 0) r += minSlotCount;
    if (r != 0) uniqueShifts.insert({source.ct, r});
  }

  // If the number of distinct rotation offsets is <= log2(N_slots), direct
  // depth-1 rotation + plaintext masking uses at most `uniqueShifts` rotations
  // at multiplicative depth 1, vs VVE's multi-stage log2(N) rotations and
  // ~2·log2(N) masks.
  int64_t maxDirectRotations =
      static_cast<int64_t>(std::ceil(std::log2(minSlotCount)));

  // Auto-select between naive shift network and Vos-Vos-Erkin shift network.
  // We may add a pass option to allow explicitly selecting VosVosErkin vs.
  // naive shift network strategy.
  if (static_cast<int64_t>(uniqueShifts.size()) <= maxDirectRotations) {
    resultNodes =
        naiveShiftNetwork(ciphertexts, mapping, minSlotCount, dagElemType);
  } else {
    ShiftScheme scheme = shiftNetworks.findShiftScheme(mapping);
    assert(!scheme.rotationGroups.empty() &&
           "Shift network must have at least one group");
    resultNodes = vosVosErkinShiftNetwork(ciphertexts, mapping, scheme,
                                          minSlotCount, dagElemType);
  }

  kernel::IRMaterializingVisitor visitor(singleCiphertextType);
  auto resultVectors = visitor.process(resultNodes, b);
  std::vector<Value> result;
  result.reserve(resultVectors.size());
  for (const auto& vec : resultVectors) {
    result.push_back(vec[0]);
  }

  // Finally, recombine with tensor.insert_slice into input tensor
  Value current = op.getInput();
  for (int64_t i = 0; i < numCiphertexts; i++) {
    auto one = b.getIndexAttr(1);
    SmallVector<OpFoldResult> offsets = {b.getIndexAttr(i), b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {one,
                                       b.getIndexAttr(tensorTy.getDimSize(1))};
    SmallVector<OpFoldResult> strides = {one, one};
    current = tensor::InsertSliceOp::create(b, op.getLoc(), result[i], current,
                                            offsets, sizes, strides);
  }

  op.replaceAllUsesWith(current);
  op.erase();
  return success();
}

struct ImplementShiftNetwork
    : impl::ImplementShiftNetworkBase<ImplementShiftNetwork> {
  using ImplementShiftNetworkBase::ImplementShiftNetworkBase;

  void runOnOperation() override {
    VosVosErkinShiftNetworks shiftNetworks;
    llvm::DenseMap<MappingCacheKey, Mapping> mappingCache;
    getOperation()->walk([&](RemapOp op) {
      if (failed(convertRemapOp(op, shiftNetworks, mappingCache))) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
