#include <cstddef>
#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/convolution/convolution_testlib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(ConvolutionTest, RunTest) {
  auto cryptoContext = convolution__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      convolution__configure_crypto_context(cryptoContext, secretKey);

  // %arg0: tensor<1x1x4x4xf32> {secret.secret}
  // ct is a [4,4] input matrix that is flattened row-major
  // [
  //  [0 1 2 3]
  //  [4 5 6 7]
  //  [8 9 10 11]
  //  [12 13 14 15]
  // ]
  std::vector<float> arg0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      arg0.push_back((float)(j + i * 4));
    }
  }

  // %filter: tensor<2x1x3x3xf32> is 2 flattened 3x3 matrices.
  // pt is a [3, 3] input filter that is packed row major
  // [
  //  [0 1 2]
  //  [3 4 5]
  //  [6 7 8]
  // ]
  std::vector<float> filters;
  for (int f = 0; f < 2; f++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        filters.push_back((float)(j + i * 3));
      }
    }
  }

  // expected is the result of the conv 2d row major, which should be two
  // repeated 2x2
  // [
  //  [ 258 294]
  //  [ 402 438]
  // ]
  std::vector<float> expected = {258.0, 294.0, 402.0, 438.0,
                                 258.0, 294.0, 402.0, 438.0};

  auto arg0Encrypted =
      convolution__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto outputEncrypted = convolution(cryptoContext, arg0Encrypted, filters);
  auto actual =
      convolution__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  ASSERT_EQ(actual.size(), expected.size());

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-3);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
