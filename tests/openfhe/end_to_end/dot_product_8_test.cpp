#include <cstdint>
#include <vector>

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
#include "tests/openfhe/end_to_end/dot_product_8_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(DotProduct8Test, RunTest) {
  // TODO(#661): Generate a helper function to set up CryptoContext based on
  // what is used in the generated code.
  CCParams<CryptoContextBGVRNS> parameters;
  parameters.SetMultiplicativeDepth(2);
  parameters.SetPlaintextModulus(65537);
  CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
  cryptoContext->Enable(PKE);
  cryptoContext->Enable(KEYSWITCH);
  cryptoContext->Enable(LEVELEDSHE);

  KeyPair<DCRTPoly> keyPair;
  keyPair = cryptoContext->KeyGen();
  cryptoContext->EvalMultKeyGen(keyPair.secretKey);
  cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, 4, 7});

  int32_t n = cryptoContext->GetCryptoParameters()
                  ->GetElementParams()
                  ->GetCyclotomicOrder() /
              2;
  int16_t arg0Vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t arg1Vals[8] = {2, 3, 4, 5, 6, 7, 8, 9};
  int64_t expected = 240;

  // TODO(#645): support cyclic repetition in add-client-interface
  std::vector<int16_t> arg0;
  std::vector<int16_t> arg1;
  arg0.reserve(n);
  arg1.reserve(n);

  for (int i = 0; i < n; ++i) {
    arg0.push_back(arg0Vals[i % 8]);
    arg1.push_back(arg1Vals[i % 8]);
  }

  auto arg0Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg0, keyPair.publicKey);
  auto arg1Encrypted =
      dot_product__encrypt__arg1(cryptoContext, arg1, keyPair.publicKey);
  auto outputEncrypted =
      dot_product(cryptoContext, arg0Encrypted, arg1Encrypted);
  auto actual = dot_product__decrypt__result0(cryptoContext, outputEncrypted,
                                              keyPair.secretKey);

  EXPECT_EQ(expected, actual);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
