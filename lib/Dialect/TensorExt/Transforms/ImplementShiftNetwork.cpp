#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/RotationGroupKernel.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/Graph/Graph.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

#define DEBUG_TYPE "implement-shift-network"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_IMPLEMENTSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

SmallVector<int64_t> defaultShiftOrder(int64_t n) {
  SmallVector<int64_t> result;
  int64_t maxLog2 = APInt(64, n).getActiveBits();
  if (isPowerOfTwo(n)) maxLog2 -= 1;
  for (int64_t i = 0; i < maxLog2; i++) result.push_back(1 << i);
  return result;
}

// Convert an input->output index mapping to a canonical left-shift amount for
// a given tensor size.
// Example: 1 -> 13 with a 64-size tensor should produce a rotation of 52
// Example: 13 -> 1 with a 64-size tensor should produce a rotation of 12
inline int64_t normalizeShift(int64_t input, int64_t output,
                              int64_t tensorSize) {
  int64_t shift = (output - input) % tensorSize;
  shift = -shift;  // Account for leftward rotations
  if (shift < 0) {
    shift += tensorSize;
  }
  return shift;
}

int64_t ShiftStrategy::getVirtualShift(const CtSlot& source,
                                       const CtSlot& target) const {
  int64_t sourceIndex = source.ct * ciphertextSize + source.slot;
  int64_t targetIndex = target.ct * ciphertextSize + target.slot;
  return normalizeShift(sourceIndex, targetIndex, virtualCiphertextSize);
}

void ShiftStrategy::evaluate(const Mapping& mapping) {
  // First compute the virtual shifts needed for each source slot
  SmallVector<SourceShift> sourceShifts;
  sourceShifts.reserve(mapping.size());
  for (const MappingEntry& entry : mapping) {
    int64_t shift = getVirtualShift(entry.source, entry.target);
    sourceShifts.push_back({entry.source, shift});
  }

  // Compute the corresponding table of positions after each rotation,
  // akin to the table in Figure 3 of the Vos-Vos-Erkin paper, including the
  // first column of values that have not yet been rotated.
  rounds.reserve(shiftOrder.size() + 1);
  ShiftRound initialRound;
  for (const SourceShift& ss : sourceShifts) {
    initialRound.positions[ss] = ss.source;
    initialRound.rotationAmount = 0;
  }
  rounds.push_back(initialRound);

  for (auto rotationAmount : shiftOrder) {
    auto lastRoundPositions = rounds.back().positions;
    DenseMap<SourceShift, CtSlot> currentRoundPosns;

    for (const SourceShift& key : sourceShifts) {
      assert(lastRoundPositions.contains(key) &&
             "Expected to find source in last round positions");
      CtSlot currentPos = lastRoundPositions[key];
      int64_t currentVirtualSlot =
          currentPos.ct * ciphertextSize + currentPos.slot;

      CtSlot nextPosition = currentPos;
      if (rotationAmount & key.shift) {
        currentVirtualSlot =
            (currentVirtualSlot - rotationAmount + virtualCiphertextSize) %
            virtualCiphertextSize;
        nextPosition = CtSlot{currentVirtualSlot / ciphertextSize,
                              currentVirtualSlot % ciphertextSize};
      }
      currentRoundPosns[key] = nextPosition;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After rotation " << rotationAmount << ":\n";
      for (const auto& [ss, pos] : currentRoundPosns) {
        llvm::dbgs() << "  (" << ss.source.ct << "," << ss.source.slot << ")["
                     << ss.shift << "] -> (" << pos.ct << "," << pos.slot << ")"
                     << "\n";
      }
    });

    rounds.push_back({currentRoundPosns, rotationAmount});
  }
}

ShiftScheme VosVosErkinShiftNetworks::findShiftScheme(
    const Mapping& mapping, ArrayRef<int64_t> shiftOrder) {
  FrozenVector<int64_t> frozenShiftOrder(shiftOrder);
  CacheKey cacheKey = std::make_pair(mapping, frozenShiftOrder);
  if (schemeCache.count(cacheKey)) {
    return schemeCache[cacheKey];
  }

  // TODO(#2256): try many shift orders and pick the best
  ShiftStrategy strategy(mapping.getCiphertextSize(),
                         mapping.getNumCiphertexts(), shiftOrder);
  strategy.evaluate(mapping);

  // Create a graph whose vertices are the input indices to permute, and
  // whose edges are conflicts: an edge being present means the two indices
  // cannot participate in the same rotation group.
  graph::UndirectedGraph<CtSlot> conflictGraph;
  for (const MappingEntry& entry : mapping) {
    conflictGraph.addVertex(entry.source);
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

void populateMappingFromLayoutAttr(const NewLayoutAttr& layoutAttr,
                                   Mapping& mapping) {
  PointPairCollector collector(2, 2);
  enumeratePoints(layoutAttr.getIntegerRelation(), collector);
  for (const auto& [source, target] : collector.points) {
    mapping.add(CtSlot{source[0], source[1]}, CtSlot{target[0], target[1]});
  }

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

LogicalResult convertPermuteOp(PermuteOp op,
                               VosVosErkinShiftNetworks& shiftNetworks) {
  LLVM_DEBUG(llvm::dbgs() << "Converting layout op: " << op << "\n");
  ImplicitLocOpBuilder b(op.getLoc(), op.getContext());
  RankedTensorType tensorTy = op.getInput().getType();
  // Since this pass may only run after convert-to-ciphertext-semantics, the
  // input must be a 2D tensor of (ct, slot) shape.
  int64_t numCiphertexts = tensorTy.getDimSize(0);
  int64_t ciphertextSize = tensorTy.getDimSize(1);
  auto singleCiphertextType =
      RankedTensorType::get({ciphertextSize}, tensorTy.getElementType());
  // Populate the mapping with (source, target) pairs
  // This require enumerating over the relation for the op
  Mapping mapping(ciphertextSize, numCiphertexts);
  if (auto layoutAttr = dyn_cast<NewLayoutAttr>(op.getPermutation())) {
    populateMappingFromLayoutAttr(layoutAttr, mapping);
  } else if (auto denseElementsAttr =
                 dyn_cast<DenseIntElementsAttr>(op.getPermutation())) {
    populateMappingFromDenseElementsAttr(denseElementsAttr, mapping);
  } else {
    return op.emitOpError()
           << "requires permutation attribute to be either NewLayoutAttr or "
              "DenseIntElementsAttr";
  }

  ShiftScheme scheme = shiftNetworks.findShiftScheme(mapping);
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

  auto resultNodes =
      implementShiftNetwork(ciphertexts, mapping, scheme, ciphertextSize);

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
    getOperation()->walk([&](PermuteOp op) {
      if (failed(convertPermuteOp(op, shiftNetworks))) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
