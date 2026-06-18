#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Polynomial/NTT.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

TEST(NTTTest, ForwardNTTAndInverseINTT64) {
  uint32_t n = 8;
  uint64_t q = 65537;

  llvm::APInt qAp(64, q);
  std::optional<llvm::APInt> rootOfUnityOpt = findPrimitive2nthRoot(qAp, n);
  ASSERT_TRUE(rootOfUnityOpt.has_value());
  uint64_t rootOfUnity = rootOfUnityOpt->getZExtValue();

  std::vector<uint64_t> coeffs = {431,  3414, 1234, 7845,
                                  2145, 7415, 5471, 8452};
  std::vector<uint64_t> originalCoeffs = coeffs;

  nttInPlace(coeffs, q, rootOfUnity);
  bool changed = false;
  for (size_t i = 0; i < n; ++i) {
    if (coeffs[i] != originalCoeffs[i]) {
      changed = true;
      break;
    }
  }
  EXPECT_TRUE(changed) << "NTT did not modify coefficients";

  inttInPlace(coeffs, q, rootOfUnity);
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(coeffs[i], originalCoeffs[i]) << "Mismatch at index " << i;
  }
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
