#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/conv_1d/conv_1d_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(Conv1DTest, RunTest) {
  auto cryptoContext = conv_1d__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = conv_1d__configure_crypto_context(cryptoContext, secretKey);

  // ct is a length 8 input vector
  std::vector<float> m = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};

  // pt is a length 4 input filter
  std::vector<float> filter = {1.0f, -1.0f, 0.0f, 1.0f};

  // expected is the result of the conv 1d row major, which should be a 5
  std::vector<float> expected = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

  auto ctEncrypted =
      conv_1d__encrypt__arg0(cryptoContext, m, keyPair.publicKey);

  auto result = conv_1d(cryptoContext, ctEncrypted, filter);

  auto actual =
      conv_1d__decrypt__result0(cryptoContext, result, keyPair.secretKey);

  ASSERT_EQ(actual.size(), expected.size());

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-3);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
