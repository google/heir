#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
// This is the function we want to call from LLVM
void _mlir_ciface_tensor_extract(StridedMemRefType<int64_t>* res,
                                 StridedMemRefType<int16_t>* arg);
}

TEST(AddTest, Test1) {
  int16_t arg[1024] = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t expected = 1;
  StridedMemRefType<int64_t> res;
  StridedMemRefType<int16_t> input{arg, arg, 0, 1024, 1};
  _mlir_ciface_tensor_extract(&res, &input);
  int64_t result = *(res.data);
  EXPECT_EQ(result, expected);
  free(res.basePtr);
}
