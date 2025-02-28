#include <cstdint>
#include <vector>

#include "gmock/gmock.h"                               // from @googletest
#include "gtest/gtest.h"                               // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/constants.h"                 // from @openfhe
#include "src/pke/include/cryptocontext-fwd.h"         // from @openfhe
#include "src/pke/include/encoding/plaintext-fwd.h"    // from @openfhe
#include "src/pke/include/gen-cryptocontext.h"         // from @openfhe
#include "src/pke/include/key/keypair.h"               // from @openfhe
#include "src/pke/include/scheme/bgvrns/gen-cryptocontext-bgvrns-params.h"  // from @openfhe
#include "src/pke/include/scheme/bgvrns/gen-cryptocontext-bgvrns.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/bgv/ciphertext_plaintext_ops/ciphertext_plaintext_ops_lib.h"

using namespace lbcrypto;
using ::testing::ElementsAre;

namespace mlir {
namespace heir {
namespace openfhe {

TEST(CiphertextPlaintextOpsTest, TestInput1) {
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
  std::vector<int64_t> vectorOfInts3 = {3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Plaintext plaintext3 = cryptoContext->MakePackedPlaintext(vectorOfInts3);
  std::vector<int64_t> vectorOfInts4 = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
  Plaintext plaintext4 = cryptoContext->MakePackedPlaintext(vectorOfInts4);

  auto ciphertext1 = cryptoContext->Encrypt(keyPair.publicKey, plaintext1);

  // Computes [(ciphertext1 + plaintext2) - plaintext3] * plaintext4
  auto ciphertextActual = test_ciphertext_plaintext_ops(
      cryptoContext, ciphertext1, plaintext2, plaintext3, plaintext4);

  Plaintext plaintextActual;
  cryptoContext->Decrypt(keyPair.secretKey, ciphertextActual, &plaintextActual);
  auto actual = plaintextActual->GetPackedValue();
  actual.resize(12);

  EXPECT_THAT(actual, ElementsAre(1, 4, 3, 8, 5, 12, 7, 16, 9, 20, 11, 24));
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
