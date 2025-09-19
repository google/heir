#ifndef LIB_UTILS_ADT_FROZENVECTOR_H_
#define LIB_UTILS_ADT_FROZENVECTOR_H_

#include <cstddef>

#include "llvm/include/llvm/ADT/DenseMapInfo.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/Hashing.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"   // from @llvm-project

namespace mlir {
namespace heir {

// A FrozenVector is a wrapper around a SmallVector, immutable after the
// initial creation, and paired with a DenseMapInfo specialization so it can be
// used as the key in a DenseMap.
template <typename T, unsigned N = 4>
class FrozenVector {
 public:
  FrozenVector() = default;
  FrozenVector(const llvm::SmallVector<T, N>& other) : vector(other) {}
  FrozenVector(llvm::ArrayRef<T> other) : vector(other) {}
  FrozenVector(llvm::SmallVector<T, N>&& other) : vector(std::move(other)) {}

  using iterator = typename llvm::SmallVector<T, N>::const_iterator;

  iterator begin() const { return vector.begin(); }
  iterator end() const { return vector.end(); }
  size_t size() const { return vector.size(); }
  bool empty() const { return vector.empty(); }
  const T& operator[](size_t idx) const { return vector[idx]; }
  const T& front() const { return vector.front(); }
  const T& back() const { return vector.back(); }

  operator llvm::ArrayRef<T>() const { return vector; }

  bool operator==(const FrozenVector& other) const {
    return vector == other.vector;
  }

 private:
  llvm::SmallVector<T, N> vector;
};

}  // namespace heir
}  // namespace mlir

namespace llvm {

// DenseMapInfo specialization to enable use as DenseMap key
template <typename T, unsigned N>
struct DenseMapInfo<::mlir::heir::FrozenVector<T, N>> {
  static ::mlir::heir::FrozenVector<T, N> getEmptyKey() {
    return ::mlir::heir::FrozenVector<T, N>();
  }

  static ::mlir::heir::FrozenVector<T, N> getTombstoneKey() {
    llvm::SmallVector<T, N> tombstone;
    tombstone.push_back(DenseMapInfo<T>::getTombstoneKey());
    return ::mlir::heir::FrozenVector<T, N>(std::move(tombstone));
  }

  static unsigned getHashValue(const ::mlir::heir::FrozenVector<T, N>& val) {
    return hash_combine_range(val.begin(), val.end());
  }

  static bool isEqual(const ::mlir::heir::FrozenVector<T, N>& lhs,
                      const ::mlir::heir::FrozenVector<T, N>& rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // LIB_UTILS_ADT_FROZENVECTOR_H_
