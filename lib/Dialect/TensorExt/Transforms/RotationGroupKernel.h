#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_ROTATIONGROUPKERNEL_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_ROTATIONGROUPKERNEL_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "lib/Dialect/TensorExt/Transforms/ShiftScheme.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

using kernel::AbstractValue;
using kernel::ArithmeticDagNode;

// Helper to get bit width from DagType
inline unsigned getBitWidth(const kernel::DagType& dagType) {
  if (std::holds_alternative<kernel::FloatType>(dagType.type_variant)) {
    return std::get<kernel::FloatType>(dagType.type_variant).bitWidth;
  } else if (std::holds_alternative<kernel::IntegerType>(
                 dagType.type_variant)) {
    return std::get<kernel::IntegerType>(dagType.type_variant).bitWidth;
  } else if (std::holds_alternative<kernel::FloatTensorType>(
                 dagType.type_variant)) {
    return std::get<kernel::FloatTensorType>(dagType.type_variant).bitWidth;
  } else if (std::holds_alternative<kernel::IntTensorType>(
                 dagType.type_variant)) {
    return std::get<kernel::IntTensorType>(dagType.type_variant).bitWidth;
  }
  llvm_unreachable("Unsupported DagType variant");
}

// Helper to create a tensor DagType from an element type and shape
inline kernel::DagType makeTensorType(const kernel::DagType& elemType,
                                      const std::vector<int64_t>& shape) {
  if (std::holds_alternative<kernel::FloatType>(elemType.type_variant)) {
    unsigned bitWidth =
        std::get<kernel::FloatType>(elemType.type_variant).bitWidth;
    return kernel::DagType::floatTensor(bitWidth, shape);
  } else if (std::holds_alternative<kernel::IntegerType>(
                 elemType.type_variant)) {
    unsigned bitWidth =
        std::get<kernel::IntegerType>(elemType.type_variant).bitWidth;
    return kernel::DagType::intTensor(bitWidth, shape);
  }
  llvm_unreachable("Expected scalar DagType");
}

//  Apply a virtual rotation to a real list of ciphertexts.
//
//  A virtual ciphertext is a flattening of a list of ciphertexts. When this
//  materializes to a set of rotations of the real ciphertexts, we need to
//  track the movement of slots between ciphertexts, and decompose the virtual
//  rotation into a set of real rotations and extra masks.
//
//  For example, we have to deal with ciphertexts which are rotated in such a
//  way that they overlap two subsequent ciphertexts in the larger "virtual"
//  ciphertext. E.g. if we have size 8 and two slots 3, 7 are rotated left by
//  -2:
//
//   ct0: . . . x . . . y
//   ct1: . . . . . . . .
//
//  then after their rotation if the desired target for slot 7 is ct1 slot 2,
//  we have the following reality
//
//   ct0: . y . . . x . .
//   ct1: . . . . . . . .
//
//  and we need to mask the position of y to add it to ct1, while masking out x
//  to keep it with ct0.
template <typename T>
std::enable_if_t<
    std::is_base_of<AbstractValue, T>::value,
    SmallVector<std::optional<std::shared_ptr<ArithmeticDagNode<T>>>>>
applyVirtualRotation(ArrayRef<std::shared_ptr<ArithmeticDagNode<T>>> input,
                     int64_t rotation,
                     const std::vector<std::vector<double>>& rotateMasks,
                     int64_t ciphertextSize, kernel::DagType elemType) {
  using NodeTy = ArithmeticDagNode<T>;
  using ValueTy = std::shared_ptr<NodeTy>;
  int64_t numCiphertexts = input.size();

  // We need to identify the (possibly two) target ciphertexts for each input
  // ciphertext that was rotated.
  //
  // If there is only one target---i.e., if the rotation was exactly the power
  // of two matching a multiple of the ciphertext size---we can update the
  // target with the rotated ciphertexts and be done.
  if (rotation % ciphertextSize == 0) {
    SmallVector<ValueTy> masked;
    masked.reserve(numCiphertexts);
    auto tensorType =
        makeTensorType(elemType, {1, static_cast<int64_t>(ciphertextSize)});
    for (const auto& [ct, mask] : llvm::zip(input, rotateMasks)) {
      // Eagerly skip masking if possible
      auto [allZero, allOne] = allZeroAllOne(mask);
      if (allZero) {
        masked.push_back(NodeTy::splat(0.0, tensorType));
      } else if (allOne) {
        masked.push_back(ct);
      } else {
        masked.push_back(
            NodeTy::mul(ct, NodeTy::constantTensor(mask, tensorType)));
      }
    }

    int64_t ciphertextShift = rotation / ciphertextSize;
    SmallVector<std::optional<ValueTy>> result;
    result.reserve(numCiphertexts);
    // We are left-rotating, so ciphertext `source` maps to `target = source -
    // ciphertextShift`
    for (int64_t target = 0; target < numCiphertexts; target++) {
      int64_t source = target + ciphertextShift + numCiphertexts;
      source = source % numCiphertexts;
      result.push_back(masked[source]);
    }
    return result;
  }

  // If there are two targets, we need to add additional masks at the split.
  // Note we are rotating left, so the split is the slot whose rotated position
  // is zero.
  //
  // Nb., there is a choice here:
  //
  //  1. Mask first, then rotate together, then mask twice to separate the
  //     two targets.
  //  2. Split mask first, mask twice, then rotate twice.
  //
  // Option (1) requires one rotation, but three ct-pt muls (depth 2), and
  // option (2) requires two rotations, but only two ct-pt muls (depth 1). Not
  // sure which is better. This func implements (2).
  SmallVector<std::optional<ValueTy>> results;
  results.resize(numCiphertexts);

  int64_t minSlot = 0;
  int64_t maxSlot = ciphertextSize - 1;
  int64_t boundarySlot = rotation;

  for (int64_t ctIndex = 0; ctIndex < numCiphertexts; ctIndex++) {
    const ValueTy& ct = input[ctIndex];
    const std::vector<double>& mask = rotateMasks[ctIndex];

    // Determine the two ciphertext targets for each input ciphertext. Rotating
    // left, so ciphertext zero wraps around to numCiphertexts-1. Easiest to do
    // it in the virtual coordinate system.
    //
    // Fist compute target1 and target2, the ciphertext indices that the rotated
    // ciphertext will straddle.
    int64_t virtualN = numCiphertexts * ciphertextSize;
    int64_t minVirtual = ctIndex * ciphertextSize + minSlot;
    int64_t maxVirtual = ctIndex * ciphertextSize + maxSlot;
    int64_t minRotated = (minVirtual - rotation + virtualN) % virtualN;
    int64_t maxRotated = (maxVirtual - rotation + virtualN) % virtualN;
    int64_t target1 = minRotated / ciphertextSize;
    int64_t target2 = maxRotated / ciphertextSize;

    assert((target1 + 1) % numCiphertexts == target2 &&
           "Expected targets to be adjacent mod numCiphertexts");

    // This handles the case of a single ciphertext, where we do not need an
    // extra split mask.
    if (target1 == target2) {
      // Eagerly skip masking if possible
      auto [allZero, allOne] = allZeroAllOne(mask);
      auto tensorType =
          makeTensorType(elemType, {1, static_cast<int64_t>(ciphertextSize)});
      if (allZero) {
        results[target1] = NodeTy::splat(0.0, tensorType);
      } else if (allOne) {
        results[target1] = NodeTy::leftRotate(ct, rotation);
      } else {
        results[target1] = NodeTy::leftRotate(
            NodeTy::mul(ct, NodeTy::constantTensor(mask, tensorType)),
            rotation);
      }
      continue;
    }

    // Split each of the input masks into two masks, one for the pre-split and
    // one for the post-split.
    std::vector<double> mask1(ciphertextSize, 0);
    std::vector<double> mask2(ciphertextSize, 0);
    for (int64_t i = 0; i < ciphertextSize; i++) {
      if (i < boundarySlot) {
        mask1[i] = mask[i];
      } else {
        mask2[i] = mask[i];
      }
    }

    // std::cout << "  mask1: ";
    // for (double v : mask1) {
    //   std::cout << v << " ";
    // }
    // std::cout << "\n";
    // std::cout << "  mask2: ";
    // for (double v : mask2) {
    //   std::cout << v << " ";
    // }
    // std::cout << "\n";

    // Apply the split masks to the input and rotate
    auto tensorType =
        makeTensorType(elemType, {1, static_cast<int64_t>(ciphertextSize)});
    std::optional<ValueTy> rotated1;
    {
      auto [allZero, _] = allZeroAllOne(mask1);
      if (allZero) {
        rotated1 = std::nullopt;
      } else {
        ValueTy masked1 =
            NodeTy::mul(ct, NodeTy::constantTensor(mask1, tensorType));
        rotated1 = NodeTy::leftRotate(masked1, rotation);
      }
    }

    std::optional<ValueTy> rotated2;
    {
      auto [allZero, _] = allZeroAllOne(mask2);
      if (allZero) {
        rotated2 = std::nullopt;
      } else {
        ValueTy masked2 =
            NodeTy::mul(ct, NodeTy::constantTensor(mask2, tensorType));
        rotated2 = NodeTy::leftRotate(masked2, rotation);
      }
    }

    if (rotated1.has_value()) {
      if (results[target1].has_value()) {
        results[target1] = NodeTy::add(*results[target1], *rotated1);
      } else {
        results[target1] = *rotated1;
      }
    }

    if (rotated2.has_value()) {
      if (results[target2].has_value()) {
        results[target2] = NodeTy::add(*results[target2], *rotated2);
      } else {
        results[target2] = *rotated2;
      }
    }
  }

  return results;
}

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 SmallVector<std::shared_ptr<ArithmeticDagNode<T>>>>
rotateOneGroup(const Mapping& mapping, ArrayRef<T> initialCiphertexts,
               ArrayRef<SourceShift> sourceShifts, ArrayRef<ShiftRound> rounds,
               const RotationGroup& group, int64_t ciphertextSize,
               kernel::DagType elemType) {
  using NodeTy = ArithmeticDagNode<T>;
  using ValueTy = std::shared_ptr<NodeTy>;

  int64_t numCiphertexts = initialCiphertexts.size();
  SmallVector<ValueTy> current;
  for (const T& ct : initialCiphertexts) {
    current.push_back(NodeTy::leaf(ct));
  }

  for (const auto& [roundNum, round] : llvm::enumerate(rounds)) {
    if (roundNum == 0) continue;

    // std::cout << "Round " << roundNum << " with rotation amount "
    //           << round.rotationAmount << "\n";

    // Need two masks, one to select the sources in this group that need to
    // be rotated, and one to preserve the values at fixed positions.
    SmallVector<CtSlot> rotatePositions;
    SmallVector<CtSlot> fixedPositions;
    for (const SourceShift& ss : sourceShifts) {
      if (!group.contains(ss.source)) continue;
      CtSlot currentPos = rounds[roundNum - 1].positions.at(ss);
      if (ss.shift & round.rotationAmount) {
        rotatePositions.push_back(currentPos);
      } else {
        fixedPositions.push_back(currentPos);
      }
    }

    // std::cout << "  Rotate positions: ";
    // for (CtSlot ctSlot : rotatePositions) {
    //   std::cout << "(" << ctSlot.ct << "," << ctSlot.slot << ") ";
    // }
    // std::cout << "\n";
    // std::cout << "  Fixed positions: ";
    // for (CtSlot ctSlot : fixedPositions) {
    //   std::cout << "(" << ctSlot.ct << "," << ctSlot.slot << ") ";
    // }
    // std::cout << "\n";

    SmallVector<std::vector<double>> fixedMasks(
        numCiphertexts, std::vector<double>(ciphertextSize, 0.0));
    for (CtSlot ctSlot : fixedPositions) {
      fixedMasks[ctSlot.ct][ctSlot.slot] = 1;
    }

    // skip masking if possible
    SmallVector<std::optional<ValueTy>> fixedCurrent;
    fixedCurrent.reserve(numCiphertexts);
    auto tensorType =
        makeTensorType(elemType, {1, static_cast<int64_t>(ciphertextSize)});
    for (const auto& [ct, fixedMask] : llvm::zip(current, fixedMasks)) {
      auto [allZero, allOne] = allZeroAllOne(fixedMask);
      if (allZero) {
        fixedCurrent.push_back(std::nullopt);
      } else if (allOne) {
        fixedCurrent.push_back(ct);
      } else {
        ValueTy mask = NodeTy::constantTensor(fixedMask, tensorType);
        fixedCurrent.push_back(NodeTy::mul(ct, mask));
      }
    }

    SmallVector<std::optional<ValueTy>> rotatedCurrent(numCiphertexts);
    rotatedCurrent.reserve(numCiphertexts);
    if (!rotatePositions.empty()) {
      std::vector<std::vector<double>> rotateMasks(
          numCiphertexts, std::vector<double>(ciphertextSize, 0.0));
      for (CtSlot ctSlot : rotatePositions) {
        rotateMasks[ctSlot.ct][ctSlot.slot] = 1;
      }

      // std::cout << "  Rotate masks:\n";
      // for (const auto& mask : rotateMasks) {
      //   for (double v : mask) {
      //     std::cout << v << " ";
      //   }
      //   std::cout << "\n";
      // }
      rotatedCurrent =
          applyVirtualRotation(ArrayRef<ValueTy>(current), round.rotationAmount,
                               rotateMasks, ciphertextSize, elemType);
    }

    // Combine the rotated and fixed parts to form the new current.
    for (int64_t i = 0; i < numCiphertexts; i++) {
      std::optional<ValueTy> fixed = fixedCurrent[i];
      std::optional<ValueTy> rotated = rotatedCurrent[i];
      if (!fixed.has_value() && !rotated.has_value()) continue;

      if (!fixed.has_value()) {
        current[i] = *rotated;
      } else if (!rotated.has_value()) {
        current[i] = *fixed;
      } else {
        current[i] = NodeTy::add(*fixed, *rotated);
      }
    }
  }

  // A set of indices that constitute the ciphertexts in this group
  // that are the final targets of some source.
  DenseSet<int64_t> finalTargetCiphertexts;
  for (const auto& [target, source] : mapping.getTargetToSource()) {
    if (group.contains(source)) {
      finalTargetCiphertexts.insert(target.ct);
    }
  }

  // Add up the results, zeroing out any ciphertexts that are not the final
  // target of some rotation, as they contain partially-shifted and fixed
  // values from middle rounds of this group.
  auto tensorType =
      makeTensorType(elemType, {1, static_cast<int64_t>(ciphertextSize)});
  for (int64_t i = 0; i < numCiphertexts; i++) {
    if (!finalTargetCiphertexts.contains(i)) {
      current[i] = NodeTy::splat(0.0, tensorType);
    }
  }

  return current;
}

template <typename T>
std::enable_if_t<
    std::is_base_of<AbstractValue, T>::value,
    SmallVector<SmallVector<std::shared_ptr<ArithmeticDagNode<T>>>>>
implementRotationGroups(SmallVector<T>& ciphertexts, const Mapping& mapping,
                        const ShiftScheme& scheme, int64_t ciphertextSize,
                        kernel::DagType elemType) {
  using NodeTy = ArithmeticDagNode<T>;
  using ValueTy = std::shared_ptr<NodeTy>;

  auto rotationGroups = scheme.rotationGroups;
  SmallVector<SmallVector<ValueTy>> groupResults;
  [[maybe_unused]] int groupIndex = 0;
  for (const RotationGroup& group : rotationGroups) {
    // Compute the subset of SourceShifts needed for this group
    SmallVector<SourceShift> sourceShifts;
    for (const auto& [target, source] : mapping.getTargetToSource()) {
      if (group.contains(source)) {
        int64_t shift = scheme.strategy.getVirtualShift(source, target);
        sourceShifts.push_back({source, shift});
      }
    }

    // std::cout << "Processing group " << groupIndex++ << " with "
    //           << sourceShifts.size() << " sources:\n";
    // for (const SourceShift& ss : sourceShifts) {
    //   std::cout << "  (" << ss.source.ct << "," << ss.source.slot << ") shift
    //   "
    //             << ss.shift << "\n";
    // }

    SmallVector<ValueTy> perGroupResult = rotateOneGroup(
        mapping, ArrayRef<T>(ciphertexts), sourceShifts,
        scheme.strategy.getRounds(), group, ciphertextSize, elemType);
    groupResults.push_back(perGroupResult);
  }

  return groupResults;
}

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 SmallVector<std::shared_ptr<ArithmeticDagNode<T>>>>
implementShiftNetwork(SmallVector<T>& ciphertexts, const Mapping& mapping,
                      const ShiftScheme& scheme, int64_t ciphertextSize,
                      kernel::DagType elemType) {
  using NodeTy = ArithmeticDagNode<T>;
  using ValueTy = std::shared_ptr<NodeTy>;
  SmallVector<SmallVector<ValueTy>> groupResults = implementRotationGroups(
      ciphertexts, mapping, scheme, ciphertextSize, elemType);

  // Add all the per-group results together
  SmallVector<ValueTy> summedResults = groupResults[0];
  summedResults.resize(ciphertexts.size());
  for (const SmallVector<ValueTy>& groupResult :
       llvm::drop_begin(groupResults)) {
    for (int i = 0; i < summedResults.size(); i++) {
      summedResults[i] = NodeTy::add(summedResults[i], groupResult[i]);
    }
  }

  return summedResults;
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_ROTATIONGROUPKERNEL_H_
