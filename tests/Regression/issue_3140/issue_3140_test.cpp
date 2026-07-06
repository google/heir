#include <cstdint>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Regression/issue_3140/issue_3140.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(Issue3140Test, RunTest) {
  auto cryptoContext = issue_3140__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      issue_3140__configure_crypto_context(cryptoContext, secretKey);

  int16_t arg0 = 2;
  int16_t arg1 = 3;
  int64_t expected = 7;

  auto arg0Encrypted =
      issue_3140__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted =
      issue_3140__encrypt__arg1(cryptoContext, arg1, publicKey);
  auto outputEncrypted =
      issue_3140(cryptoContext, arg0Encrypted, arg1Encrypted);
  auto actual =
      issue_3140__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_EQ(expected, actual);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
