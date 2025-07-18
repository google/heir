#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_

#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"      // from @llvm-project

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

  SmallVector<ShiftRound> getRounds() const;

  // Run the shifting strategy and populate the `rounds` member variable.
  void evaluate(const Permutation &permutation, const RotationGroup &group);

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
  ArrayRef<RotationGroup> computeShiftNetwork(const Permutation &permutation);

  int64_t getCiphertextSize() const;

 private:
  int64_t ciphertextSize;
  DenseMap<Permutation, llvm::SmallVector<RotationGroup>> rotationGroups;
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_IMPLEMENTSHIFTNETWORK_H_
