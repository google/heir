#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/bicyclic_matmul_pt/bicyclic_matmul_pt_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BicyclicMatmulPtOpenFHERobustTest, RunTest) {
  auto cryptoContext = bicyclic_matmul_pt__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      bicyclic_matmul_pt__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0(13 * 14, 0.0);
  std::vector<float> arg1(14 * 16, 0.0);

  // A[i][j] = i + j
  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 14; ++j) {
      arg0[i * 14 + j] = (i + j) / 100.0;
    }
  }

  // B[j][k] = j - k
  for (int j = 0; j < 14; ++j) {
    for (int k = 0; k < 16; ++k) {
      arg1[j * 16 + k] = (j - k) / 100.0;
    }
  }

  // Precompute expected C[i][k] from actual arg0 and arg1 vectors
  std::vector<float> expected(13 * 16, 0.0);
  for (int i = 0; i < 13; ++i) {
    for (int k = 0; k < 16; ++k) {
      for (int j = 0; j < 14; ++j) {
        expected[i * 16 + k] += arg0[i * 14 + j] * arg1[j * 16 + k];
      }
    }
  }

  auto arg0Encrypted =
      bicyclic_matmul_pt__encrypt__arg0(cryptoContext, arg0, publicKey);

  auto outputEncrypted = bicyclic_matmul_pt(cryptoContext, arg0Encrypted, arg1);

  auto actual = bicyclic_matmul_pt__decrypt__result0(
      cryptoContext, outputEncrypted, secretKey);

  float errorThreshold = 1e-2;
  for (int i = 0; i < 13 * 16; ++i) {
    EXPECT_NEAR(expected[i], actual[i], errorThreshold);
  }
}

TEST(BicyclicMatmulPtOpenFHERobustTest, RunPreprocessedTest) {
  auto cryptoContext = bicyclic_matmul_pt__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      bicyclic_matmul_pt__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0(13 * 14, 0.0);
  std::vector<float> arg1(14 * 16, 0.0);

  // A[i][j] = i + j
  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 14; ++j) {
      arg0[i * 14 + j] = (i + j) / 100.0;
    }
  }

  // B[j][k] = j - k
  for (int j = 0; j < 14; ++j) {
    for (int k = 0; k < 16; ++k) {
      arg1[j * 16 + k] = (j - k) / 100.0;
    }
  }

  // Precompute expected C[i][k] from actual arg0 and arg1 vectors
  std::vector<float> expected(13 * 16, 0.0);
  for (int i = 0; i < 13; ++i) {
    for (int k = 0; k < 16; ++k) {
      for (int j = 0; j < 14; ++j) {
        expected[i * 16 + k] += arg0[i * 14 + j] * arg1[j * 16 + k];
      }
    }
  }

  auto arg0Encrypted =
      bicyclic_matmul_pt__encrypt__arg0(cryptoContext, arg0, publicKey);

  // Preprocess plaintext weight diagonals
  auto preprocessedDiagonals =
      bicyclic_matmul_pt__preprocessing(cryptoContext, arg1);

  // Run using preprocessed diagonals
  auto outputEncrypted = bicyclic_matmul_pt__preprocessed(
      cryptoContext, arg0Encrypted, arg1, preprocessedDiagonals);

  auto actual = bicyclic_matmul_pt__decrypt__result0(
      cryptoContext, outputEncrypted, secretKey);

  float errorThreshold = 1e-2;
  for (int i = 0; i < 13 * 16; ++i) {
    EXPECT_NEAR(expected[i], actual[i], errorThreshold);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
