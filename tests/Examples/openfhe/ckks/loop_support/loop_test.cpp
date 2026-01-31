#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "gmock/gmock.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/loop_support/loop_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

using ::testing::Pointwise;
using ::testing::DoubleNear;

TEST(LoopTest, RunTest) {
  auto cryptoContext = loop__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = loop__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0 = {0.,         0.14285714, 0.28571429, 0.42857143,
                             0.57142857, 0.71428571, 0.85714286, 1.};
  std::vector<float> expected = {-1.,         -1.16666629, -1.39989342,
                                 -1.74687019, -2.29543899, -3.19507837,
                                 -4.66914279, -7.};

  auto arg0Encrypted = loop__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto outputEncrypted = loop(cryptoContext, secretKey, arg0Encrypted);
  auto actual =
      loop__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_THAT(actual, Pointwise(DoubleNear(1e-03), expected));
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
