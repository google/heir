#include <cstdint>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/bgv/box_blur/box_blur_16x16_lib.h"

using ::testing::ContainerEq;

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BoxBlurTest, TestInput1) {
  // needs to be large enough to accommodate overflow in the plaintext space
  // 786433 is the smallest prime p above 2**17 for which (p-1) / 65536 is an
  // integer.
  auto cryptoContext = box_blur__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = box_blur__configure_crypto_context(cryptoContext, secretKey);

  std::vector<int16_t> input;
  std::vector<int16_t> expected;
  input.reserve(256);
  expected.reserve(256);

  for (int i = 0; i < 256; ++i) {
    input.push_back(i);
  }

  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int16_t sum = 0;
      for (int di = -1; di < 2; ++di) {
        for (int dj = -1; dj < 2; ++dj) {
          int index = (row * 16 + col + di * 16 + dj) % 256;
          if (index < 0) index += 256;
          sum += input[index];
        }
      }
      expected.push_back(sum);
    }
  }

  auto inputEncrypted =
      box_blur__encrypt__arg0(cryptoContext, input, keyPair.publicKey);
  auto outputEncrypted = box_blur(cryptoContext, inputEncrypted);
  auto actual = box_blur__decrypt__result0(cryptoContext, outputEncrypted,
                                           keyPair.secretKey);

  EXPECT_THAT(actual, ContainerEq(expected));
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
