#include <cstdint>
#include <vector>

#include "gtest/gtest.h"              // from @googletest
#include "src/pke/include/openfhe.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/dot_product_8f_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

// TODO(#891): support other schemes besides BGV in add-client-interface
CiphertextT dot_product__encrypt__arg0(CryptoContextT v16,
                                       std::vector<double> v17,
                                       PublicKeyT v18) {
  int32_t n =
      v16->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  std::vector<double> outputs;
  outputs.reserve(n);
  for (int i = 0; i < n; ++i) {
    outputs.push_back(v17[i % v17.size()]);
  }
  const auto& v19 = v16->MakeCKKSPackedPlaintext(outputs);
  const auto& v20 = v16->Encrypt(v18, v19);
  return v20;
}

double dot_product__decrypt__result0(CryptoContextT v26, CiphertextT v27,
                                     PrivateKeyT v28) {
  PlaintextT v29;
  v26->Decrypt(v28, v27, &v29);
  double v30 = v29->GetCKKSPackedValue()[0].real();
  return v30;
}

TEST(DotProduct8FTest, RunTest) {
  auto cryptoContext = dot_product__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      dot_product__configure_crypto_context(cryptoContext, secretKey);

  std::vector<double> arg0 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<double> arg1 = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  double expected = 2.4 + 0.1;

  // TODO(#891): support other schemes besides BGV in add-client-interface
  auto arg0Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg1, publicKey);
  auto outputEncrypted =
      dot_product(cryptoContext, arg0Encrypted, arg1Encrypted);
  auto actual =
      dot_product__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-3);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
