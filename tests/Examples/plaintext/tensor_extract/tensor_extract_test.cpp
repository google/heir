#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
// This is the function we want to call from LLVM
void _mlir_ciface_tensor_extract(StridedMemRefType<int64_t, 2>* res,
                                 StridedMemRefType<int64_t, 2>* arg);

void _mlir_ciface_tensor_extract__encrypt__arg0(
    StridedMemRefType<int64_t, 2>* result, StridedMemRefType<int16_t>* arg);

// Note that our output is i16 but decoder takes i64 due to mod-arith
int16_t _mlir_ciface_tensor_extract__decrypt__result0(
    StridedMemRefType<int64_t, 2>* arg);
}

TEST(AddTest, Test1) {
  int16_t arg[1024] = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t expected = 1;

  StridedMemRefType<int64_t, 2> encArg0{};
  StridedMemRefType<int16_t> input{arg, arg, 0, 1024, 1};
  _mlir_ciface_tensor_extract__encrypt__arg0(&encArg0, &input);

  StridedMemRefType<int64_t, 2> res{};
  _mlir_ciface_tensor_extract(&res, &encArg0);

  int16_t result = _mlir_ciface_tensor_extract__decrypt__result0(&res);
  EXPECT_EQ(result, expected);
  free(res.basePtr);
  free(encArg0.basePtr);
}
