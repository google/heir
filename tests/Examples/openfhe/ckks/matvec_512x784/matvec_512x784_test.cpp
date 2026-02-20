#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "tests/Examples/openfhe/ckks/matvec_512x784/matvec_512x784_lib.h"

TEST(Matvec512x784Test, RunTest) {
  auto cryptoContext = matvec__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = matvec__configure_crypto_context(cryptoContext, secretKey);

  int cols = 784;
  int rows = 512;
  std::vector<float> arg0(cols, 0.1);

  float expected = 78.4;

  auto ct0 = matvec__encrypt__arg0(cryptoContext, arg0, publicKey);

  auto resultCt = matvec(cryptoContext, ct0);

  auto result = matvec__decrypt__result0(cryptoContext, resultCt, secretKey);

  float errorThreshold = 0.1;
  for (int i = 0; i < rows; i++) {
    EXPECT_NEAR(expected, result[i], errorThreshold);
  }
}
