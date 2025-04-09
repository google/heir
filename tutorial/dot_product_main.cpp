#include <cstdint>
#include <vector>

// #include "src/pke/include/openfhe.h" // from @openfhe
#include <openfhe.h>

#include "heir_output.h"

int main(int argc, char *argv[]) {
  CryptoContext<DCRTPoly> cryptoContext =
      dot_product__generate_crypto_context();

  KeyPair<DCRTPoly> keyPair;
  keyPair = cryptoContext->KeyGen();

  cryptoContext =
      dot_product__configure_crypto_context(cryptoContext, keyPair.secretKey);

  std::vector<float> arg0 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<float> arg1 = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  float expected = 2.4 + 0.1;

  auto arg0Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg0, keyPair.publicKey);
  auto arg1Encrypted =
      dot_product__encrypt__arg1(cryptoContext, arg1, keyPair.publicKey);
  auto outputEncrypted =
      dot_product(cryptoContext, arg0Encrypted, arg1Encrypted);
  auto actual = dot_product__decrypt__result0(cryptoContext, outputEncrypted,
                                              keyPair.secretKey);

  std::cout << "Expected: " << expected << "\n";
  std::cout << "Actual: " << actual << "\n";

  return 0;
}
