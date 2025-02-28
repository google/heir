#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/bgv/simple_sum/simple_sum_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BinopsTest, TestInput1) {
  auto cryptoContext = simple_sum__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      simple_sum__configure_crypto_context(cryptoContext, secretKey);

  std::vector<int16_t> input = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  int64_t expected = 16 * 33;

  auto inputEncrypted =
      simple_sum__encrypt__arg0(cryptoContext, input, keyPair.publicKey);
  auto outputEncrypted = simple_sum(cryptoContext, inputEncrypted);
  auto actual = simple_sum__decrypt__result0(cryptoContext, outputEncrypted,
                                             keyPair.secretKey);

  EXPECT_EQ(expected, actual);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
