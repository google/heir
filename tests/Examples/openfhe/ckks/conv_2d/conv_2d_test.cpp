#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/conv_2d/conv_2d_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(Conv2DTest, RunTest) {
  auto cryptoContext = conv_2d__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = conv_2d__configure_crypto_context(cryptoContext, secretKey);

  // ct is a [3, 3] input matrix that is flattened row-major
  std::vector<float> m(9, 0.1f);

  // pt is a [2, 2] input filter that is packed row major
  std::vector<float> filter(4, 0.1f);

  // expected is the result of the conv 2d row major, which should be a 2x2
  std::vector<float> expected = {0.04f, 0.04f, 0.04f, 0.04f};

  auto ctEncrypted =
      conv_2d__encrypt__arg0(cryptoContext, m, keyPair.publicKey);

  auto result = conv_2d(cryptoContext, ctEncrypted, filter);

  auto actual =
      conv_2d__decrypt__result0(cryptoContext, result, keyPair.secretKey);

  ASSERT_EQ(actual.size(), expected.size());

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-3);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
