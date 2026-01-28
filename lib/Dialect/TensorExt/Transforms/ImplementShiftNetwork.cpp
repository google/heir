#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/RotationGroupKernel.h"
#include "lib/Dialect/TensorExt/Transforms/ShiftScheme.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/Graph/Graph.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
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

ShiftSchemeResult VosVosErkinShiftNetworks::findShiftScheme(
    const Mapping& mapping, ArrayRef<int64_t> shiftOrder) {
  CacheKey cacheKey = makeCacheKey(mapping, shiftOrder);
  if (schemeCache.count(cacheKey)) {
    return ShiftSchemeResult{schemeCache[cacheKey], mapping};
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

    auto posns = round.positions;
    for (auto it1 = posns.begin(); it1 != posns.end(); ++it1) {
      for (auto it2 = std::next(it1); it2 != posns.end(); ++it2) {
        const SourceShift& ss1 = it1->first;
        const SourceShift& ss2 = it2->first;
        if (ss1.source != ss2.source && it1->second == it2->second) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Round " << roundNum << ": collision between " << "{"
                     << ss1.source.ct << "," << ss1.source.slot << "}"
                     << " and " << "{" << ss2.source.ct << ","
                     << ss2.source.slot << "}" << " at " << "{"
                     << it1->second.ct << "," << it1->second.slot << "}\n");
          conflictGraph.addEdge(ss1.source, ss2.source);
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

  // Cleaning subroutine, identify if overlapping sources for targets
  Mapping cleanedMapping = mapping;
  auto targetToSources = mapping.getTargetToSources();
  bool hasMultiSourceTargets =
      std::any_of(targetToSources.begin(), targetToSources.end(),
                  [](const auto& kv) { return kv.second.size() > 1; });
  LLVM_DEBUG({
    llvm::dbgs() << "Has overlapping sources for targets - "
                 << hasMultiSourceTargets << "\n";
  });
  if (hasMultiSourceTargets) {
    // Create a full conflict graph
    graph::UndirectedGraph<CtSlot> fullConflictGraph;
    for (const auto& [target, sources] : targetToSources) {
      for (const auto& ctmatch : sources) {
        fullConflictGraph.addVertex(ctmatch.source);
      }
    }
    SmallVector<int64_t> defaultTestShiftOrder = defaultShiftOrder(
        mapping.getCiphertextSize() * mapping.getNumCiphertexts());

    ShiftStrategy fullConflictStrategy =
        evaluateShiftStrategy(mapping, defaultTestShiftOrder, true);

    for (const auto& [roundNum, round] :
         llvm::enumerate(fullConflictStrategy.getRounds())) {
      if (roundNum == 0) continue;
      auto posns = round.positions;
      for (auto it1 = posns.begin(); it1 != posns.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != posns.end(); ++it2) {
          const SourceShift& ss1 = it1->first;
          const SourceShift& ss2 = it2->first;
          if (ss1.source != ss2.source && it1->second == it2->second) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Round " << roundNum << ": collision between " << "{"
                       << ss1.source.ct << "," << ss1.source.slot << "}"
                       << " and " << "{" << ss2.source.ct << ","
                       << ss2.source.slot << "}" << " at " << "{"
                       << it1->second.ct << "," << it1->second.slot << "}\n");
            fullConflictGraph.addEdge(ss1.source, ss2.source);
          }
        }
      }
    }

    struct SourceInfo {
      int numConflicts;
      int numInserts;
    };

    std::unordered_map<CtSlot, SourceInfo> sourceMap;

    for (CtSlot vertex : fullConflictGraph.getVertices()) {
      sourceMap[vertex] = SourceInfo{
          static_cast<int>(fullConflictGraph.edgesIncidentTo(vertex).size()),
          0};
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Full Conflict graph:\n";
      for (CtSlot vertex : fullConflictGraph.getVertices()) {
        llvm::dbgs() << vertex.ct << "," << vertex.slot << " <-> {";
        for (CtSlot neighbor : fullConflictGraph.edgesIncidentTo(vertex)) {
          llvm::dbgs() << neighbor.ct << "," << neighbor.slot << "; ";
        }
        llvm::dbgs() << "}\n";
      }
    });

    // Create seeds for tests
    static std::mt19937 masterRng(std::random_device{}());
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<std::uint32_t> seedDist;

    // Store best mapping
    int minNumRotations = std::numeric_limits<int>::max();
    Mapping bestMapping = mapping;
    graph::UndirectedGraph<CtSlot> bestConflictGraph;

    // Copy the target to Sources for testing
    std::vector<std::pair<CtSlot, SmallVector<Mapping::CtMatch>*>> allTargets;
    allTargets.reserve(targetToSources.size());
    for (auto& entry : targetToSources) {
      allTargets.emplace_back(entry.first, &entry.second);
    }

    // Run 50 tests
    for (int i = 0; i < 50; ++i) {
      // track source numberConflicts , inserts
      std::unordered_map<CtSlot, SourceInfo> testSourceMap = sourceMap;

      // create test mapping and conflictgraph
      Mapping testMapping = mapping;
      graph::UndirectedGraph<CtSlot> testConflictGraph;

      // shuffle targets and fill
      std::uint32_t trialSeed = seedDist(masterRng);
      rng.seed(trialSeed);
      std::shuffle(allTargets.begin(), allTargets.end(), rng);

      for (auto& [slot, matchesPtr] : allTargets) {
        SmallVector<Mapping::CtMatch>& matches = *matchesPtr;
        std::vector<std::pair<CtSlot, SourceInfo*>> candidates;
        candidates.reserve(matches.size());

        // Get and sort sources by 1) inserts , 2) conflicts ascending
        for (auto& match : matches) {
          CtSlot src = match.source;
          auto it = testSourceMap.find(src);
          candidates.emplace_back(src, &it->second);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                    const SourceInfo& sa = *a.second;
                    const SourceInfo& sb = *b.second;
                    if (sa.numInserts != sb.numInserts)
                      return sa.numInserts < sb.numInserts;
                    return sa.numConflicts < sb.numConflicts;
                  });

        LLVM_DEBUG({
          llvm::dbgs() << "Candidates after sort for target " << slot.ct << ","
                       << slot.slot << ":\n";
          for (const auto& entry : candidates) {
            const CtSlot& src = entry.first;
            const SourceInfo& info = *entry.second;
            llvm::dbgs() << "  src=(" << src.ct << "," << src.slot
                         << "), numInserts=" << info.numInserts
                         << ", numConflicts=" << info.numConflicts << "\n";
          }
        });

        // Pick any source with lowest at random and save choice
        const SourceInfo& best = *candidates.front().second;
        int minInserts = best.numInserts;
        int minConflicts = best.numConflicts;

        std::vector<size_t> bestIdx;
        bestIdx.reserve(candidates.size());

        // any source with lowest
        for (size_t i = 0; i < candidates.size(); ++i) {
          const SourceInfo& info = *candidates[i].second;
          if (info.numInserts == minInserts &&
              info.numConflicts == minConflicts)
            bestIdx.push_back(i);
          else
            break;
        }

        // pick and save choice
        std::uniform_int_distribution<size_t> dist(0, bestIdx.size() - 1);
        size_t chosenIndex = bestIdx[dist(rng)];

        CtSlot chosen = candidates[chosenIndex].first;
        SourceInfo& info = *candidates[chosenIndex].second;
        info.numInserts += 1;
        testMapping.add(chosen, slot);
        testConflictGraph.addVertex(chosen);
      }

      // Create test strategy, fill edges
      ShiftStrategy testStrategy =
          evaluateShiftStrategy(testMapping, defaultTestShiftOrder);
      for (const auto& [roundNum, round] :
           llvm::enumerate(testStrategy.getRounds())) {
        if (roundNum == 0) continue;
        auto posns = round.positions;
        for (auto it1 = posns.begin(); it1 != posns.end(); ++it1) {
          for (auto it2 = std::next(it1); it2 != posns.end(); ++it2) {
            const SourceShift& ss1 = it1->first;
            const SourceShift& ss2 = it2->first;
            if (ss1.source != ss2.source && it1->second == it2->second) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Round " << roundNum << ": collision between "
                         << "{" << ss1.source.ct << "," << ss1.source.slot
                         << "}" << " and " << "{" << ss2.source.ct << ","
                         << ss2.source.slot << "}" << " at " << "{"
                         << it1->second.ct << "," << it1->second.slot << "}\n");
              testConflictGraph.addEdge(ss1.source, ss2.source);
            }
          }
        }
      }

      // Find number of rotations needed
      graph::GreedyGraphColoring<CtSlot> testColorer;
      std::unordered_map<CtSlot, int> testColoring =
          testColorer.color(testConflictGraph);

      SmallVector<RotationGroup> testResultRotationGroups;
      testResultRotationGroups.reserve(5);

      for (const auto& entry : testColoring) {
        CtSlot source = entry.first;
        int64_t color = entry.second;
        if (color >= static_cast<int64_t>(testResultRotationGroups.size()))
          testResultRotationGroups.resize(color + 1);
        testResultRotationGroups[color].insert(source);
      }

      // Update best number rotations & mapping
      int testNumberRotations =
          static_cast<int>(testResultRotationGroups.size());
      if (testNumberRotations < minNumRotations) {
        minNumRotations = testNumberRotations;
        bestMapping = testMapping;
        bestConflictGraph = testConflictGraph;
      }
    }

    cleanedMapping = bestMapping;
    conflictGraph = bestConflictGraph;

    LLVM_DEBUG({
      llvm::dbgs() << "Cleaning subroutine finished. "
                   << "Best num rotations = " << minNumRotations << "\n";

      llvm::dbgs() << "Best mapping targetToSource:\n";
      for (const auto& [target, source] : bestMapping.getTargetToSource()) {
        llvm::dbgs() << "  target=(" << target.ct << "," << target.slot << ")"
                     << " <- source=(" << source.ct << "," << source.slot
                     << ")\n";
      }

      llvm::dbgs() << "Conflict graph after cleaning:\n";
      for (CtSlot vertex : conflictGraph.getVertices()) {
        llvm::dbgs() << "  " << vertex.ct << "," << vertex.slot << " <-> {";
        for (CtSlot neighbor : conflictGraph.edgesIncidentTo(vertex)) {
          llvm::dbgs() << neighbor.ct << "," << neighbor.slot << "; ";
        }
        llvm::dbgs() << "}\n";
      }
    });
  }

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
  return ShiftSchemeResult{schemeCache[cacheKey], cleanedMapping};
}

ShiftSchemeResult VosVosErkinShiftNetworks::findBestShiftScheme(
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

    std::ranges::shuffle(shiftOrder.begin(), shiftOrder.end(), g);

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
    const Mapping& mapping, ArrayRef<int64_t> shiftOrder, bool useSources) {
  CacheKey cacheKey = makeCacheKey(mapping, shiftOrder);
  if (strategyCache.count(cacheKey)) {
    return strategyCache[cacheKey];
  }

  ShiftStrategy strategy(mapping.getCiphertextSize(),
                         mapping.getNumCiphertexts(), shiftOrder);
  strategy.evaluate(mapping, useSources);
  strategyCache[cacheKey] = strategy;
  return strategy;
}

VosVosErkinShiftNetworks::CacheKey VosVosErkinShiftNetworks::makeCacheKey(
    const Mapping& mapping, ArrayRef<int64_t> shiftOrder) {
  FrozenVector<int64_t> frozenShiftOrder(shiftOrder);
  return std::make_pair(mapping, frozenShiftOrder);
}

void populateMappingFromLayoutAttr(const LayoutAttr& layoutAttr,
                                   Mapping& mapping) {
  PointPairCollector collector(2, 2);
  enumeratePoints(layoutAttr.getIntegerRelation(), collector);

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

LogicalResult convertRemapOp(RemapOp op,
                             VosVosErkinShiftNetworks& shiftNetworks) {
  LLVM_DEBUG(llvm::dbgs() << "Converting layout op: " << op << "\n");
  ImplicitLocOpBuilder b(op.getLoc(), op.getContext());
  RankedTensorType tensorTy = op.getInput().getType();
  // Since this pass may only run after convert-to-ciphertext-semantics, the
  // input must be a 2D tensor of (ct, slot) shape.
  int64_t numCiphertexts = tensorTy.getDimSize(0);
  int64_t ciphertextSize = tensorTy.getDimSize(1);
  auto singleCiphertextType =
      RankedTensorType::get({1, ciphertextSize}, tensorTy.getElementType());
  // Populate the mapping with (source, target) pairs
  // This require enumerating over the relation for the op
  Mapping mapping(ciphertextSize, numCiphertexts);
  if (auto layoutAttr = dyn_cast<LayoutAttr>(op.getPermutation())) {
    populateMappingFromLayoutAttr(layoutAttr, mapping);
  } else if (auto denseElementsAttr =
                 dyn_cast<DenseIntElementsAttr>(op.getPermutation())) {
    populateMappingFromDenseElementsAttr(denseElementsAttr, mapping);
  } else {
    return op.emitOpError()
           << "requires permutation attribute to be either LayoutAttr or "
              "DenseIntElementsAttr";
  }

  auto shiftSchemeResult = shiftNetworks.findShiftScheme(mapping);
  ShiftScheme& scheme = shiftSchemeResult.scheme;
  Mapping& cleanedMapping = shiftSchemeResult.cleanedMapping;
  auto rotationGroups = scheme.rotationGroups;

  assert(!rotationGroups.empty() &&
         "Shift network must have at least one group");

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

  auto resultNodes = implementShiftNetwork(ciphertexts, cleanedMapping, scheme,
                                           ciphertextSize);

  kernel::IRMaterializingVisitor visitor(b, singleCiphertextType);
  std::vector<Value> result = visitor.process(resultNodes);

  // Finally, recombine with an empty tensor + tensor.insert_slice
  Value combinedResult = tensor::EmptyOp::create(
      b, {numCiphertexts, ciphertextSize}, tensorTy.getElementType());
  Value current = combinedResult;
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
    getOperation()->walk([&](RemapOp op) {
      if (failed(convertRemapOp(op, shiftNetworks))) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
