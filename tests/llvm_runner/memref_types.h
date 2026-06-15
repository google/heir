#ifndef TESTS_LLVM_RUNNER_MEMREF_TYPES_H_
#define TESTS_LLVM_RUNNER_MEMREF_TYPES_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Adapted from CRunnerUtils from MLIR

// StridedMemRef descriptor type with static rank.
template <typename T, std::size_t N = 1>
struct StridedMemRefType {
  T* basePtr = nullptr;
  T* data = nullptr;
  int64_t offset = 0;
  int64_t sizes[N] = {0};
  int64_t strides[N] = {0};

  StridedMemRefType() = default;

  StridedMemRefType(T* basePtr, T* data, int64_t offset, int64_t size,
                    int64_t stride)
    requires(N == 1)
      : basePtr{basePtr},
        data{data},
        offset{offset},
        sizes{size},
        strides{stride} {}

  StridedMemRefType(T* basePtr, T* data, int64_t offset,
                    const int64_t (&sizes_)[N], const int64_t (&strides_)[N])
      : basePtr{basePtr}, data{data}, offset{offset} {
    std::copy_n(sizes_, N, sizes);
    std::copy_n(strides_, N, strides);
  }
};

// Unranked MemRef
template <typename T>
struct UnrankedMemRefType {
  int64_t rank = 0;
  void* descriptor = nullptr;
};

#endif  // TESTS_LLVM_RUNNER_MEMREF_TYPES_H_
