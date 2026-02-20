#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers
#include "tests/Examples/openfhe/ckks/validate_lower/validate_lower_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(ValidateLowerTest, RunTest) {
  auto cryptoContext = test_validate_lower__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      test_validate_lower__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

  auto arg0Encrypted =
      test_validate_lower__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto outputEncrypted = test_validate_lower(cryptoContext, arg0Encrypted);
  // Just test that no assertions are hit in the callbacks defined in
  // debug_helper.h
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
