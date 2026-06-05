#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/batchnorm_sigmoid/batchnorm_sigmoid_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BatchNormSigmoidTest, RunTest) {
  auto cryptoContext = batchnorm_sigmoid__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      batchnorm_sigmoid__configure_crypto_context(cryptoContext, secretKey);

  float x = 0.5;
  float mean = 0.1;
  float var = 0.2;
  float gamma = 1.0;
  float beta = 0.0;
  float expected = 0.709798;  // 1 / (1 + exp(-0.8944))

  auto xEncrypted =
      batchnorm_sigmoid__encrypt__arg0(cryptoContext, x, publicKey);

  auto outputEncrypted =
      batchnorm_sigmoid(cryptoContext, xEncrypted, mean, var, gamma, beta);

  auto actual = batchnorm_sigmoid__decrypt__result0(cryptoContext,
                                                    outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-1);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
