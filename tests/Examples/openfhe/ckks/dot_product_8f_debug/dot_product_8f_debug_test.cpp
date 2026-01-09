#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/dot_product_8f_debug/dot_product_8f_debug_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(DotProduct8FTest, RunTest) {
  auto cryptoContext = dot_product__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      dot_product__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<float> arg1 = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  float expected = 2.4 + 0.1;

  auto arg0Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg1, publicKey);
  auto outputEncrypted =
      dot_product(cryptoContext, secretKey, arg0Encrypted, arg1Encrypted);
  auto actual =
      dot_product__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-3);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
