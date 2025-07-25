#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"   // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"          // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

#define DEBUG_TYPE "implement-shift-network"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_IMPLEMENTSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

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

SmallVector<ShiftRound> ShiftStrategy::getRounds() const { return rounds; }

void ShiftStrategy::evaluate(const Permutation &permutation,
                             const RotationGroup &group) {
  int64_t ciphertextSize = permutation.size();

  // Stores the amount that each ciphertext index is shifted left. The
  // RotationGroup might be a subset of indices, so we have to populate the
  // entire set of shifts with zeros except possibly for those in the
  // RotationGroup (some of those may also be zero if they're fixed points of
  // the permutation).
  SmallVector<int64_t> shifts;
  shifts.resize(ciphertextSize, 0);
  for (int64_t index : group) {
    shifts[index] = normalizeShift(index, permutation[index], ciphertextSize);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Shifts for permutation: ";
    for (int64_t shift : shifts) {
      llvm::dbgs() << shift << " ";
    }
    llvm::dbgs() << "\n";
  });

  // The identity permutation is a base case where no shifts are applied.
  Permutation identityPerm = FrozenVector<int64_t>(identity(ciphertextSize));
  for (int64_t rotationAmount = 1; rotationAmount < ciphertextSize;
       rotationAmount <<= 1) {
    ShiftRound round;
    ArrayRef<int64_t> lastRoundPositions =
        !rounds.empty() ? ArrayRef<int64_t>(rounds.back().positions)
                        : identityPerm;
    int inputIndex = 0;
    for (int64_t shift : shifts) {
      // The bit is set, implying we would rotate by 2**bit in this round
      if (shift & rotationAmount) {
        // subtract because we are left-shifting by a positive amount
        int64_t dest = lastRoundPositions[inputIndex] - rotationAmount;
        if (dest < 0) {
          dest += ciphertextSize;  // wrap around the bottom of the vector
        }
        round.positions.push_back(dest);
        round.rotatedIndices.push_back(inputIndex);
      } else {
        // Otherwise the value is unchanged from last round
        round.positions.push_back(lastRoundPositions[inputIndex]);
      }
      ++inputIndex;
    }
    round.rotationAmount = rotationAmount;
    rounds.push_back(round);
    LLVM_DEBUG({
      llvm::dbgs() << "Round " << rotationAmount << ": ";
      int inputIndex = 0;
      for (int64_t index : round.positions) {
        llvm::dbgs() << inputIndex++ << ": " << index << ", ";
      }
      llvm::dbgs() << "\n";
      llvm::dbgs() << "Indices affected: ";
      for (int64_t index : round.rotatedIndices) {
        llvm::dbgs() << index << " ";
      }
      llvm::dbgs() << "\n";
    });
  }
}

ArrayRef<RotationGroup> VosVosErkinShiftNetworks::computeShiftNetwork(
    const Permutation &permutation) {
  if (rotationGroups.count(permutation)) {
    return rotationGroups[permutation];
  }

  ShiftStrategy strategy;
  RotationGroup allIndices;
  for (int64_t i = 0; i < ciphertextSize; i++) {
    allIndices.insert(i);
  }
  strategy.evaluate(permutation, allIndices);

  // Create a graph whose vertices are the input indices to permute, and
  // whose edges are conflicts: an edge being present means the two indices
  // cannot participate in the same rotation group.
  graph::UndirectedGraph<int64_t> conflictGraph;
  for (int64_t i = 0; i < ciphertextSize; i++) {
    conflictGraph.addVertex(i);
  }
  for (const ShiftRound &round : strategy.getRounds()) {
    for (int64_t i = 0; i < ciphertextSize; i++) {
      for (int64_t j = i + 1; j < ciphertextSize; j++) {
        if (round.positions[i] == round.positions[j]) {
          conflictGraph.addEdge(i, j);
        }
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Conflict graph:\n";
    for (int64_t vertex : conflictGraph.getVertices()) {
      llvm::dbgs() << "  " << vertex << ": ";
      for (int64_t neighbor : conflictGraph.edgesIncidentTo(vertex)) {
        llvm::dbgs() << neighbor << " ";
      }
      llvm::dbgs() << "\n";
    }
  });

  graph::GreedyGraphColoring<int64_t> colorer;
  std::unordered_map<int64_t, int> coloring = colorer.color(conflictGraph);

  SmallVector<RotationGroup> resultRotationGroups;
  resultRotationGroups.reserve(64);
  for (const auto &entry : coloring) {
    int64_t index = entry.first;
    int64_t color = entry.second;
    if (color >= resultRotationGroups.size()) {
      resultRotationGroups.resize(color + 1);
    }
    resultRotationGroups[color].insert(index);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Splitting permutation into permutation groups:\n";
    for (int i = 0; i < resultRotationGroups.size(); i++) {
      llvm::dbgs() << "Group " << i << ": ";
      llvm::SmallVector<int64_t> group = llvm::SmallVector<int64_t>(
          resultRotationGroups[i].begin(), resultRotationGroups[i].end());
      llvm::sort(group);
      for (int64_t index : group) {
        llvm::dbgs() << index << " ";
      }
      llvm::dbgs() << "\n";
    }
  });

  rotationGroups[permutation] = resultRotationGroups;
  return rotationGroups[permutation];
}

int64_t VosVosErkinShiftNetworks::getCiphertextSize() const {
  return ciphertextSize;
}

// Create a tensor with zeros everywhere except for the indices specified in
// the input `indices` vector.
Value createMask(TypedValue<RankedTensorType> tensor,
                 const SmallVector<int64_t> &indices, IRRewriter &rewriter) {
  auto elementType = tensor.getType().getElementType();
  SmallVector<Attribute> maskAttrs(tensor.getType().getDimSize(0),
                                   rewriter.getIntegerAttr(elementType, 0));
  for (int64_t index : indices) {
    maskAttrs[index] = rewriter.getIntegerAttr(elementType, 1);
  }

  auto denseAttr = DenseElementsAttr::get(tensor.getType(), maskAttrs);
  auto constant =
      rewriter.create<arith::ConstantOp>(tensor.getLoc(), denseAttr);
  return constant.getResult();
}

Value rotateGroup(TypedValue<RankedTensorType> tensor,
                  const RotationGroup &group, int64_t ciphertextSize,
                  const Permutation &permutation, IRRewriter &rewriter) {
  std::optional<Value> result = std::nullopt;

  // Re-run the shift strategy on a single rotation group, and use the
  // rotatedIndices in each round to construct a mask and a rotation op.
  ShiftStrategy strategy;
  strategy.evaluate(permutation, group);

  [[maybe_unused]] int roundNum = 0;
  for (const ShiftRound &round : strategy.getRounds()) {
    int64_t rotationAmount = round.rotationAmount;
    if (round.rotatedIndices.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping shift by " << rotationAmount
                              << " because no inputs have that bit set\n");
      continue;
    }

    Value mask = createMask(tensor, round.rotatedIndices, rewriter);
    arith::MulIOp maskOp =
        rewriter.create<arith::MulIOp>(tensor.getLoc(), tensor, mask);
    Value rotated = rewriter.create<tensor_ext::RotateOp>(
        tensor.getLoc(), maskOp.getResult(),
        rewriter.create<arith::ConstantIntOp>(
            tensor.getLoc(), rewriter.getI32Type(), rotationAmount));

    if (result.has_value()) {
      result = rewriter.create<arith::AddIOp>(tensor.getLoc(), result.value(),
                                              rotated);
    } else {
      result = rotated;
    }
  }

  return result.has_value() ? result.value() : tensor;
}

LogicalResult convertPermuteOp(PermuteOp op,
                               VosVosErkinShiftNetworks &shiftNetworks,
                               int64_t ciphertextSize) {
  LLVM_DEBUG(llvm::dbgs() << "Converting layout op: " << op << "\n");
  IRRewriter rewriter(op.getContext());
  RankedTensorType tensorTy = op.getInput().getType();

  // Only support a 1-D tensor until sharding is supported
  if (op.getInput().getType().getRank() != 1) {
    return op.emitError("requires a one-dimensional tensor");
  }

  SmallVector<int64_t> permutation;

  // Convert the affine map to an explicit permutation, or else use the dense
  // array attr as an explicit permutation.
  auto affineMapAttr = dyn_cast<AffineMapAttr>(op.getPermutation());
  if (affineMapAttr) {
    AffineMap permutationMap = simplifyAffineMap(affineMapAttr.getValue());
    LLVM_DEBUG(llvm::dbgs()
               << "Expanding permutation from " << permutationMap << "\n");
    if (failed(makeExplicit1DMapping(permutationMap, tensorTy.getNumElements(),
                                     permutation)))
      return failure();
  } else {
    auto denseElementsAttr =
        dyn_cast<DenseIntElementsAttr>(op.getPermutation());
    if (denseElementsAttr) {
      permutation = llvm::map_to_vector(
          denseElementsAttr, [](const APInt &i) { return i.getSExtValue(); });
    } else {
      return failure();
    }
  }

  if (!isPermutation(permutation)) {
    auto diag = op.emitError(
        "expected a permutation, but got a mapping that was not a "
        "permutation.");
    Diagnostic &note = diag.attachNote() << "mapping was:\n";
    printPermutation(permutation, note);
    return diag;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "PermuteOp produces underlying permutation: ";
    printPermutation(permutation, llvm::dbgs());
  });

  FrozenVector<int64_t> permKey = FrozenVector<int64_t>(std::move(permutation));
  ArrayRef<RotationGroup> rotationGroup =
      shiftNetworks.computeShiftNetwork(permKey);
  assert(!rotationGroup.empty() &&
         "Shift network must have at least one group");

  // Process each rotation group separately with a full set of power-of-two
  // shifts. Then sum the results together.
  rewriter.setInsertionPointAfter(op);
  std::optional<Value> result = std::nullopt;
  [[maybe_unused]] int groupIndex = 0;
  for (const RotationGroup &group : rotationGroup) {
    LLVM_DEBUG(llvm::dbgs()
               << "Implementing rotations for group " << groupIndex++ << "\n");
    Value perGroupResult =
        rotateGroup(op.getInput(), group, ciphertextSize, permKey, rewriter);
    if (result.has_value())
      result =
          rewriter.create<arith::AddIOp>(op.getLoc(), *result, perGroupResult);
    else
      result = perGroupResult;
  }

  rewriter.replaceOp(op, result.value());
  return success();
}

struct ImplementShiftNetwork
    : impl::ImplementShiftNetworkBase<ImplementShiftNetwork> {
  using ImplementShiftNetworkBase::ImplementShiftNetworkBase;

  void runOnOperation() override {
    VosVosErkinShiftNetworks shiftNetworks{ciphertextSize};

    getOperation()->walk([&](PermuteOp op) {
      if (failed(convertPermuteOp(op, shiftNetworks, ciphertextSize))) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
