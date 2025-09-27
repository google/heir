#include <cstdint>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
// This is the function we want to call from LLVM
void _mlir_ciface_add(StridedMemRefType<int64_t>* res,
                      StridedMemRefType<int16_t>* arg0
                      /*, StridedMemRefType<int16_t>* arg1 */);
}

TEST(AddTest, Test1) {
  int16_t arg0[1024] = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t expected = 1;

  StridedMemRefType<int64_t> res;
  StridedMemRefType<int16_t> lhs{arg0, arg0, 0, 1024, 1};
  // StridedMemRefType<int16_t> rhs{arg1, arg1, 0, 8, 1};
  _mlir_ciface_add(&res, &lhs /*, &rhs */);

  int64_t result = *(res.data);

  EXPECT_EQ(result, expected);
}
