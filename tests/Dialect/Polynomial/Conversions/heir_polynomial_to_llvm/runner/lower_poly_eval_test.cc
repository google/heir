#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
float _mlir_ciface_test_poly_eval(float arg0);
}

TEST(LowerMulTest, TestPolyEval) {
  float result = _mlir_ciface_test_poly_eval(5.0);
  ASSERT_EQ(result, 31.0f);
}
