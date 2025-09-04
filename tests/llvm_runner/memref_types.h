#ifndef TESTS_LLVM_RUNNER_MEMREF_TYPES_H_
#define TESTS_LLVM_RUNNER_MEMREF_TYPES_H_

#include <cstdint>

// Copied from CRunnerUtils from MLIR

// StridedMemRef descriptor type with static rank.
template <typename T, int N>
struct StridedMemRefType {
  T* basePtr;
  T* data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// Unranked MemRef
template <typename T>
struct UnrankedMemRefType {
  int64_t rank;
  void* descriptor;
};

#endif  // TESTS_LLVM_RUNNER_MEMREF_TYPES_H_
