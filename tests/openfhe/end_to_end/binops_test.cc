#include <cstdint>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "openfhe.h"      // from @openfhe
#include "tests/openfhe/end_to_end/binops_lib.h"

using namespace lbcrypto;
using ::testing::ElementsAre;

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BinopsTest, TestInput1) {
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
  cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, -1, -2});

  std::vector<int64_t> vectorOfInts1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Plaintext plaintext1 = cryptoContext->MakePackedPlaintext(vectorOfInts1);
  std::vector<int64_t> vectorOfInts2 = {3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Plaintext plaintext2 = cryptoContext->MakePackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cryptoContext->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cryptoContext->Encrypt(keyPair.publicKey, plaintext2);

  // Computes (vectorOfInts1 + vectorOfInts2) * (vectorOfInts1 - vectorOfInts2)
  auto ciphertextActual = test_binops(cryptoContext, ciphertext1, ciphertext2);

  Plaintext plaintextActual;
  cryptoContext->Decrypt(keyPair.secretKey, ciphertextActual, &plaintextActual);
  auto actual = plaintextActual->GetPackedValue();
  actual.resize(12);

  EXPECT_THAT(actual, ElementsAre(-8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
