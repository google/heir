#ifndef TESTS_BENCHMARK_MEMREF_H_
#define TESTS_BENCHMARK_MEMREF_H_

#include <stdlib.h>

#include <cstdint>

namespace heir {
namespace test {

// Simple implementation of a 2-D Memref Descriptor.
// For reference, see
// https://github.com/llvm/llvm-project/blob/6ee845d2401b7f0e5f385fc0e3a8cb44afd667dc/mlir/lib/Conversion/LLVMCommon/TypeConverter.cpp#L331-L338
// Implementation was inspired by
// https://github.com/neonhuang/buddy-mlir/blob/450d9107f2b2a66d5d5f8bc76ecf69c6cee368f8/include/Interface/buddy/dip/memref.h#L27
class Memref {
 public:
  Memref(int64_t h, int64_t w, int32_t value = 0) {
    allocatedPtr = (int32_t*)malloc(sizeof(int32_t) * w * h);
    alignedPtr = allocatedPtr;

    offset = 0;
    sizes[0] = h;
    sizes[1] = w;
    strides[0] = w;
    strides[1] = 1;

    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        *pget(i, j) = value;
      }
    }
  }

  int32_t* pget(int64_t i, int64_t j) const {
    return &alignedPtr[offset + i * strides[0] + j * strides[1]];
  }

  int32_t get(int64_t i, int64_t j) const {
    return alignedPtr[offset + i * strides[0] + j * strides[1]];
  }

 private:
  int32_t* allocatedPtr;
  int32_t* alignedPtr;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

}  // namespace test
}  // namespace heir

#endif  // TESTS_BENCHMARK_MEMREF_H_
