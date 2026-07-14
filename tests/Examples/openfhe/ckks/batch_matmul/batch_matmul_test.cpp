#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/batch_matmul/batch_matmul_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BatchMatmulOpenFHERobustTest, RunTest) {
  auto cryptoContext = batch_matmul__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      batch_matmul__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0(2 * 17 * 19, 0.0);
  std::vector<float> arg1(2 * 19 * 21, 0.0);

  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 17; ++i) {
      for (int j = 0; j < 19; ++j) {
        arg0[b * 17 * 19 + i * 19 + j] = (b + i + j) / 100.0;
      }
    }
  }

  for (int b = 0; b < 2; ++b) {
    for (int j = 0; j < 19; ++j) {
      for (int k = 0; k < 21; ++k) {
        arg1[b * 19 * 21 + j * 21 + k] = (b + j - k) / 100.0;
      }
    }
  }

  std::vector<float> expected(2 * 17 * 21, 0.0);
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 17; ++i) {
      for (int k = 0; k < 21; ++k) {
        for (int j = 0; j < 19; ++j) {
          expected[b * 17 * 21 + i * 21 + k] +=
              arg0[b * 17 * 19 + i * 19 + j] * arg1[b * 19 * 21 + j * 21 + k];
        }
      }
    }
  }

  auto arg0Encrypted =
      batch_matmul__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted =
      batch_matmul__encrypt__arg1(cryptoContext, arg1, publicKey);

  auto outputEncrypted =
      batch_matmul(cryptoContext, arg0Encrypted, arg1Encrypted);

  auto actual =
      batch_matmul__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  float errorThreshold = 1e-2;
  for (int i = 0; i < 2 * 17 * 21; ++i) {
    EXPECT_NEAR(expected[i], actual[i], errorThreshold);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
