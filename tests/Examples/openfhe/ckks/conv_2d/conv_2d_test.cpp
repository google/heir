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

  // ct is a [4,4] input matrix that is flattened row-major
  // [
  //  [0 1 2 3]
  //  [4 5 6 7]
  //  [8 9 10 11]
  //  [12 13 14 15]
  // ]
  std::vector<float> m;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      m.push_back((float)(j + i * 4));
    }
  }

  // pt is a [3, 3] input filter that is packed row major
  // [
  //  [0 1 2]
  //  [3 4 5]
  //  [6 7 8]
  // ]
  std::vector<float> filter;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      filter.push_back((float)(j + i * 3));
    }
  }

  // expected is the result of the conv 2d row major, which should be a 2x2
  // [
  //  [ 258 294]
  //  [ 402 438]
  // ]
  std::vector<float> expected = {258.0, 294.0, 402.0, 438.0};

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
