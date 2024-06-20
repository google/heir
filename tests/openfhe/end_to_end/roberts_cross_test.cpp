#include <cstdint>
#include <vector>

#include "gmock/gmock.h"                               // from @googletest
#include "gtest/gtest.h"                               // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/constants.h"                 // from @openfhe
#include "src/pke/include/cryptocontext-fwd.h"         // from @openfhe
#include "src/pke/include/gen-cryptocontext.h"         // from @openfhe
#include "src/pke/include/key/keypair.h"               // from @openfhe
#include "src/pke/include/openfhe.h"                   // from @openfhe
#include "src/pke/include/scheme/bgvrns/gen-cryptocontext-bgvrns-params.h"  // from @openfhe
#include "src/pke/include/scheme/bgvrns/gen-cryptocontext-bgvrns.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/openfhe/end_to_end/roberts_cross_64x64_lib.h"

using ::testing::ContainerEq;

namespace mlir {
namespace heir {
namespace openfhe {

TEST(RobertsCrossTest, TestInput1) {
  auto cryptoContext = roberts_cross__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      roberts_cross__configure_crypto_context(cryptoContext, secretKey);

  int32_t n = cryptoContext->GetCryptoParameters()
                  ->GetElementParams()
                  ->GetCyclotomicOrder() /
              2;
  std::vector<int16_t> input;
  std::vector<int16_t> expected;
  input.reserve(n);
  expected.reserve(4096);

  // TODO(#645): support cyclic repetition in add-client-interface
  for (int i = 0; i < n; ++i) {
    input.push_back(i % 4096);
  }

  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 64; ++col) {
      // (img[x-1][y-1] - img[x][y])^2 + (img[x-1][y] - img[x][y-1])^2
      int xY = (row * 64 + col) % 4096;
      int xYm1 = (row * 64 + col - 1) % 4096;
      int xm1Y = ((row - 1) * 64 + col) % 4096;
      int xm1Ym1 = ((row - 1) * 64 + col - 1) % 4096;

      if (xYm1 < 0) xYm1 += 4096;
      if (xm1Y < 0) xm1Y += 4096;
      if (xm1Ym1 < 0) xm1Ym1 += 4096;

      int16_t v1 = (input[xm1Ym1] - input[xY]);
      int16_t v2 = (input[xm1Y] - input[xYm1]);
      int16_t sum = v1 * v1 + v2 * v2;
      expected.push_back(sum);
    }
  }

  auto inputEncrypted =
      roberts_cross__encrypt__arg0(cryptoContext, input, keyPair.publicKey);
  auto outputEncrypted = roberts_cross(cryptoContext, inputEncrypted);
  auto actual = roberts_cross__decrypt__result0(cryptoContext, outputEncrypted,
                                                keyPair.secretKey);

  EXPECT_THAT(actual, ContainerEq(expected));
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
