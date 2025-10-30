#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_SHIFTSCHEME_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_SHIFTSCHEME_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <utility>

#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/Hashing.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "llvm/include/llvm/ADT/DenseMap.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMapInfo.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"      // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace tensor_ext {

// A (ciphertext, slot) pair representing a specific slot in a specific
// ciphertext.
struct CtSlot {
  int64_t ct;
  int64_t slot;

  CtSlot(int64_t ct, int64_t slot) : ct(ct), slot(slot) {}
  CtSlot() : ct(0), slot(0) {}

  bool operator==(const CtSlot& other) const {
    return ct == other.ct && slot == other.slot;
  }

  bool operator!=(const CtSlot& other) const { return !(*this == other); }

  bool operator<(const CtSlot& other) const {
    return ct < other.ct || (ct == other.ct && slot < other.slot);
  }

  bool operator>(const CtSlot& other) const {
    return ct > other.ct || (ct == other.ct && slot > other.slot);
  }
};

inline ::llvm::hash_code hash_value(const CtSlot& obj) {
  return llvm::hash_combine(obj.ct, obj.slot);
}

struct MappingEntry {
  CtSlot source;
  CtSlot target;
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

namespace llvm {
template <>
struct DenseMapInfo<mlir::heir::tensor_ext::CtSlot> {
  static mlir::heir::tensor_ext::CtSlot getEmptyKey() {
    return mlir::heir::tensor_ext::CtSlot{-1, -1};
  }
  static mlir::heir::tensor_ext::CtSlot getTombstoneKey() {
    return mlir::heir::tensor_ext::CtSlot{-1, -2};
  }
  static unsigned getHashValue(const mlir::heir::tensor_ext::CtSlot& Val) {
    return hash_combine(Val.ct, Val.slot);
  }
  static bool isEqual(const mlir::heir::tensor_ext::CtSlot& L,
                      const mlir::heir::tensor_ext::CtSlot& R) {
    return L == R;
  }
};
}  // namespace llvm

namespace mlir {
namespace heir {
namespace tensor_ext {

// An arbitrary mapping on the slots of a set of ciphertexts.
class Mapping {
 public:
  Mapping(int64_t ciphertextSize = 1, int64_t numCiphertexts = 1)
      : ciphertextSize(ciphertextSize), numCiphertexts(numCiphertexts) {}

  size_t size() const { return entries.size(); }

  void add(CtSlot source, CtSlot target) {
    auto [it, inserted] = entries.insert({target, source});
    if (!inserted) {
      // Update the mapping if the new source is closer to the target than the
      // existing source. This will select for the closest source when there are
      // multiple valid choices for the source in the mapping but will help
      // reduce conflicts when the source and target have repetition.
      // TODO(#2350): Consider a better way to handle multiple valid sources.
      CtSlot& existingSource = it->second;
      if (getVirtualDistance(target, source) <
          getVirtualDistance(target, existingSource)) {
        entries[target] = source;
      }
    }
  }

  auto begin() const { return entries.begin(); }
  auto end() const { return entries.end(); }

  int64_t getCiphertextSize() const { return ciphertextSize; }
  int64_t getNumCiphertexts() const { return numCiphertexts; }

 private:
  int64_t ciphertextSize;
  int64_t numCiphertexts;
  // Map from target to source
  DenseMap<CtSlot, CtSlot> entries;

  int64_t getVirtualDistance(const CtSlot& lhs, const CtSlot& rhs) {
    return std::abs(lhs.ct - rhs.ct) * ciphertextSize +
           std::abs(lhs.slot - rhs.slot);
  }
};

// A group of source CtSlots to rotate together
using RotationGroup = DenseSet<CtSlot>;

// A set of ciphertexts is represented as a single large virtual ciphertext
// (flattened row-major), and so a given (ct, slot) pair is "shifted" by an
// amount that may exceed the size of a single ciphertext. The algorithm keeps
// track of the bookkeeping needed to identify the actual target ciphertext
// when materializing a RotationGroup to actual rotations.
struct SourceShift {
  CtSlot source;
  int64_t shift;  // Virtual left shift amount

  bool operator==(const SourceShift& other) const {
    return source == other.source && shift == other.shift;
  }

  bool operator!=(const SourceShift& other) const { return !(*this == other); }
};

// The ShiftStrategy class applies power-of-two shifts to each set bit in some
// order (default is LSB-to-MSB order, 1, 2, 4, 8, ...). Each shift amount is
// considered a "round" in which a group of indices are shifted together. This
// can be used both to identify conflicts for the graph coloring technique of
// Vos-Vos-Erkin, and also to construct the concrete shift network after a
// partition has been decided by Vos-Vos-Erkin.
struct ShiftRound {
  // Maps the index of the original mapping source to its current position in
  // the tensor. This may contain multiple indices mapping to the same slot due
  // to conflicts in the shifting strategy.
  DenseMap<SourceShift, CtSlot> positions;
  // The (virtual) amount rotated left
  int64_t rotationAmount;
};

/// Return the default shift order: LSB to MSB, i.e. 1, 2, 4, 8, ...
SmallVector<int64_t> defaultShiftOrder(int64_t n);

class ShiftStrategy {
 public:
  ShiftStrategy() : ShiftStrategy(1, 1) {}

  ShiftStrategy(int64_t ciphertextSize, int64_t numCiphertexts = 1,
                ArrayRef<int64_t> shiftOrder = {})
      : ciphertextSize(ciphertextSize),
        virtualCiphertextSize(numCiphertexts * ciphertextSize),
        shiftOrder(shiftOrder.empty()
                       ? defaultShiftOrder(numCiphertexts * ciphertextSize)
                       : shiftOrder) {
    assert(isPowerOfTwo(ciphertextSize) &&
           "ciphertext size must be a power of two");
  }

  // Return the
  int64_t getVirtualShift(const CtSlot& source, const CtSlot& target) const;

  SmallVector<ShiftRound> getRounds() const { return rounds; }

  // Run the shifting strategy and populate the list of rounds in the strategy
  void evaluate(const Mapping& mapping);

 private:
  int64_t ciphertextSize;
  int64_t virtualCiphertextSize;
  SmallVector<int64_t> shiftOrder;
  SmallVector<ShiftRound> rounds;
};

struct ShiftScheme {
  SmallVector<RotationGroup> rotationGroups;
  ShiftStrategy strategy;

  ShiftScheme() = default;
  ShiftScheme(SmallVector<RotationGroup> rotationGroups, ShiftStrategy strategy)
      : rotationGroups(std::move(rotationGroups)),
        strategy(std::move(strategy)) {}
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

// SourceShift, Mapping, and CtSlot needs DenseMapInfo for use in DenseMap
namespace llvm {

template <>
struct DenseMapInfo<mlir::heir::tensor_ext::SourceShift> {
  static mlir::heir::tensor_ext::SourceShift getEmptyKey() {
    return mlir::heir::tensor_ext::SourceShift{
        mlir::heir::tensor_ext::CtSlot{-1, -1}, -1};
  }
  static mlir::heir::tensor_ext::SourceShift getTombstoneKey() {
    return mlir::heir::tensor_ext::SourceShift{
        mlir::heir::tensor_ext::CtSlot{-1, -2}, -1};
  }
  static unsigned getHashValue(const mlir::heir::tensor_ext::SourceShift& Val) {
    return hash_combine(Val.source.ct, Val.source.slot, Val.shift);
  }
  static bool isEqual(const mlir::heir::tensor_ext::SourceShift& L,
                      const mlir::heir::tensor_ext::SourceShift& R) {
    return L == R;
  }
};

template <>
struct DenseMapInfo<mlir::heir::tensor_ext::Mapping> {
  static mlir::heir::tensor_ext::Mapping getEmptyKey() {
    return mlir::heir::tensor_ext::Mapping(-1, -1);
  }
  static mlir::heir::tensor_ext::Mapping getTombstoneKey() {
    return mlir::heir::tensor_ext::Mapping(-1, -2);
  }
  static unsigned getHashValue(const mlir::heir::tensor_ext::Mapping& Val) {
    unsigned hash =
        hash_combine(Val.getCiphertextSize(), Val.getNumCiphertexts());
    for (const auto& entry : Val) {
      hash ^= hash_combine(entry.second.ct, entry.second.slot, entry.first.ct,
                           entry.first.slot);
    }
    return hash;
  }
  static bool isEqual(const mlir::heir::tensor_ext::Mapping& L,
                      const mlir::heir::tensor_ext::Mapping& R) {
    if (L.getCiphertextSize() != R.getCiphertextSize() ||
        L.getNumCiphertexts() != R.getNumCiphertexts()) {
      return false;
    }
    if (L.size() != R.size()) {
      return false;
    }
    auto itL = L.begin();
    auto itR = R.begin();
    for (; itL != L.end() && itR != R.end(); ++itL, ++itR) {
      if (itL->second != itR->second || itL->first != itR->first) {
        return false;
      }
    }
    return true;
  }
};
}  // namespace llvm

namespace std {
template <>
struct hash<::mlir::heir::tensor_ext::CtSlot> {
  size_t operator()(const ::mlir::heir::tensor_ext::CtSlot& obj) const {
    return hash<int64_t>()(obj.ct) ^ (hash<int64_t>()(obj.slot) << 1);
  }
};

}  // namespace std

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_SHIFTSCHEME_H_
