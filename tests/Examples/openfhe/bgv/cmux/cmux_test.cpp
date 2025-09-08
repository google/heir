#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/bgv/cmux/cmux_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(CmuxTest, RunTest) {
  auto cryptoContext = cmux__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = cmux__configure_crypto_context(cryptoContext, secretKey);

  std::vector<int64_t> arg0 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> arg1 = {2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<bool> cond = {true, false, true, true, false, true, false, true};
  std::vector<int64_t> expected = {1, 3, 3, 4, 6, 6, 8, 8};

  for (auto i = 0; i < cond.size(); ++i) {
    auto condEncrypted = cmux__encrypt__arg2(cryptoContext, cond[i], publicKey);
    auto outputEncrypted = cmux(cryptoContext, arg0[i], arg1[i], condEncrypted);
    auto actual =
        cmux__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

    EXPECT_EQ(expected[i], actual);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
