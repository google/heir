#include <cstdint>
#include <vector>

#include "gmock/gmock.h"              // from @googletest
#include "gtest/gtest.h"              // from @googletest
#include "src/pke/include/openfhe.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/pisa/end_to_end/basic_inc.h"

using namespace lbcrypto;
using ::testing::ElementsAre;

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BasicTest, TestInput1) {
  // Test Inputs
  std::vector<int16_t> x = {1, 2, 3, 4, 5, 6, 7};
  std::vector<int16_t> y = {3, 2, 1, 4, 5, 6, 7};

  // HEIR-generated func that sets parameters and generates OpenFHE context
  auto context = basic_test__generate_crypto_context();

  auto [publicKey, secretKey] = context->KeyGen();

  // HEIR-generated funcs for encoding & encrypting of cleartext values
  auto xEnc = basic_test__encrypt__arg0(context, x, publicKey);
  auto yEnc = basic_test__encrypt__arg1(context, y, publicKey);

  // HEIR-generated func for actual computation (here: simple addition)
  auto ciphertextActual = basic_test(context, xEnc, yEnc);

  // HEIR-generated func for
  auto r = basic_test__decrypt__result0(context, ciphertextActual, secretKey);

  r.resize(x.size());
  EXPECT_THAT(r, ElementsAre(4, 4, 4, 8, 10, 12, 14));
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
