#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/bicyclic_matmul/bicyclic_matmul_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BicyclicMatmulOpenFHERobustTest, RunTest) {
  auto cryptoContext = bicyclic_matmul__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      bicyclic_matmul__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0(16 * 17, 0.0);
  std::vector<float> arg1(17 * 19, 0.0);

  // A[i][j] = i + j
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 17; ++j) {
      arg0[i * 17 + j] = i + j;
    }
  }

  // B[j][k] = j - k
  for (int j = 0; j < 17; ++j) {
    for (int k = 0; k < 19; ++k) {
      arg1[j * 19 + k] = j - k;
    }
  }

  // Precompute expected C[i][k] from actual arg0 and arg1 vectors
  std::vector<float> expected(16 * 19, 0.0);
  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 19; ++k) {
      for (int j = 0; j < 17; ++j) {
        expected[i * 19 + k] += arg0[i * 17 + j] * arg1[j * 19 + k];
      }
    }
  }

  auto arg0Encrypted =
      bicyclic_matmul__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted =
      bicyclic_matmul__encrypt__arg1(cryptoContext, arg1, publicKey);

  auto outputEncrypted =
      bicyclic_matmul(cryptoContext, arg0Encrypted, arg1Encrypted);

  auto actual = bicyclic_matmul__decrypt__result0(cryptoContext,
                                                  outputEncrypted, secretKey);

  float errorThreshold = 1e-2;
  for (int i = 0; i < 16 * 19; ++i) {
    EXPECT_NEAR(expected[i], actual[i], errorThreshold);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
