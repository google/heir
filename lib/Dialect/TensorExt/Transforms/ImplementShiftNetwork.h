#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_

#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"      // from @llvm-project

#define DEBUG_TYPE "implement-shift-network"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DECL_IMPLEMENTSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

// A permutation of 0..n-1. This vector should always have size n and contain
// each integer from 0 to n-1 exactly once.
using Permutation = FrozenVector<int64_t>;

// A group of indices to rotate together
using RotationGroup = DenseSet<int64_t>;

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

// The ShiftStrategy class applies power-of-two shifts to each set bit in
// LSB-to-MSB order, 1, 2, 4, 8, .... Each shift amount is considered a "round"
// in which a group of indices are attempted to be shifted together. This can be
// used both to identify conflicts for the graph coloring technique of
// Vos-Vos-Erkin, and also to construct the concrete shift network after a
// partition has been decided by Vos-Vos-Erkin.
struct ShiftRound {
  // Maps the index of the original input to its current position in the
  // tensor. This may contain multiple indices mapping to the same slot due to
  // conflicts in the shifting strategy.
  SmallVector<int64_t> positions;
  // The set of indices that are rotated left in this round. This can be used
  // to generate a mask to select the indices that need rotating.
  SmallVector<int64_t> rotatedIndices;
  // The amount rotated left;
  int64_t rotationAmount;
};

class ShiftStrategy {
 public:
  ShiftStrategy() = default;

  SmallVector<ShiftRound> getRounds() const { return rounds; }

  // Run the shifting strategy and populate the `rounds` member variable.
  void evaluate(const Permutation &permutation, const RotationGroup &group) {
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

 private:
  SmallVector<ShiftRound> rounds;
};

// Cf. https://www.jeremykun.com/2024/09/02/shift-networks/
// and https://link.springer.com/chapter/10.1007/978-3-031-17140-6_20
// for an explanation of the algorithm.
class VosVosErkinShiftNetworks {
 public:
  VosVosErkinShiftNetworks(int64_t ciphertextSize)
      : ciphertextSize(ciphertextSize) {}

  // Computes a partition of the slot indices of a ciphertext into
  // RotationGroups that are compatible with respect to the target permutation.
  // Each RotationGroup corresponds to a set of indices that should be rotated
  // together via power-of-two rotations.
  //
  // The returned ArrayRef is owned by this VosVosErkinShiftNetworks instance.
  // The resulting set of rotation groups are is cached, and the cache is used
  // on further calls to avoid recomputing the shift network.
  ArrayRef<RotationGroup> computeShiftNetwork(const Permutation &permutation) {
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

  int64_t getCiphertextSize() const { return ciphertextSize; }

 private:
  int64_t ciphertextSize;
  DenseMap<Permutation, llvm::SmallVector<RotationGroup>> rotationGroups;
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_
