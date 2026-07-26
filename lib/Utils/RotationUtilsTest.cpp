#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/RotationUtils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

std::vector<int64_t> sorted(const llvm::DenseSet<int64_t>& s) {
  std::vector<int64_t> v(s.begin(), s.end());
  std::sort(v.begin(), v.end());
  return v;
}

// Dense diagonals 0..127 with baby-step size bs=16: babies are r % 16 -> {1..15}
// and giants are (r/16)*16 -> {16,32,...,112}. This is the key set the emitter's
// bs (and so getRotationIndices) must produce for a dense layer.
TEST(RotationUtilsTest, DenseDiagonalsBabyStep) {
  std::vector<int32_t> diags;
  for (int i = 0; i < 128; ++i) diags.push_back(i);
  auto set = lintransRotationIndicesWithBabyStep(diags, /*slots=*/4096, /*n1=*/16);

  std::vector<int64_t> expected;
  for (int b = 1; b < 16; ++b) expected.push_back(b);
  for (int g = 16; g <= 112; g += 16) expected.push_back(g);
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(sorted(set), expected);
}

// Wrap-around layer (fc3-style): diagonals 0..127 and 4087..4095 with bs == slots
// (gs = 1, the pure-diagonal method). Every diagonal is its own baby step
// (giant = 0), so the key set is exactly the nonzero diagonal indices -- no
// giant steps spanning the empty gap.
TEST(RotationUtilsTest, WrapAroundPureDiagonal) {
  std::vector<int32_t> diags;
  for (int i = 0; i < 128; ++i) diags.push_back(i);
  for (int i = 4087; i <= 4095; ++i) diags.push_back(i);
  auto set =
      lintransRotationIndicesWithBabyStep(diags, /*slots=*/4096, /*n1=*/4096);

  std::vector<int64_t> expected;
  for (int i = 1; i < 128; ++i) expected.push_back(i);
  for (int i = 4087; i <= 4095; ++i) expected.push_back(i);
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(sorted(set), expected);
}

// The logBSGS wrapper must agree with the explicit-baby-step variant when given
// the n1 that findBestBSGSRatio selects, i.e. they are the same computation.
TEST(RotationUtilsTest, WrapperMatchesExplicitBabyStep) {
  std::vector<int32_t> diags = {0, 1, 2, 3, 4, 5, 6, 7};
  int64_t slots = 16;
  int64_t n1 = findBestBSGSRatio(diags, slots, /*logMaxRatio=*/1);
  EXPECT_EQ(sorted(lintransRotationIndices(diags, slots, /*logBSGS=*/1)),
            sorted(lintransRotationIndicesWithBabyStep(diags, slots, n1)));
}

// A baby-step size below 1 is clamped to 1: every diagonal becomes a giant step
// (baby = 0), so the key set is the nonzero diagonal indices.
TEST(RotationUtilsTest, BabyStepClampedToAtLeastOne) {
  std::vector<int32_t> diags = {0, 1, 2};
  auto set = lintransRotationIndicesWithBabyStep(diags, /*slots=*/8, /*n1=*/0);
  EXPECT_EQ(sorted(set), (std::vector<int64_t>{1, 2}));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
