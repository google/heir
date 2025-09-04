#include <cstdint>

#include "gtest/gtest.h"  // from @googletest

struct ReturnType {
  int64_t degree;
  int32_t coeff;
};

extern "C" {
void _mlir_ciface_test_leading_term(ReturnType* result);
}

TEST(LowerLeadingTermTest, TestLeadingTerm) {
  ReturnType result;
  _mlir_ciface_test_leading_term(&result);
  EXPECT_EQ(result.coeff, 2);
  EXPECT_EQ(result.degree, 10);
}
