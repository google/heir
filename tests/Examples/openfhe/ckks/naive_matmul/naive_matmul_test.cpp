#include <cstdint>
#include <ctime>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"                               // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/constants.h"                 // from @openfhe
#include "src/pke/include/cryptocontext-fwd.h"         // from @openfhe
#include "src/pke/include/gen-cryptocontext.h"         // from @openfhe
#include "src/pke/include/key/keypair.h"               // from @openfhe
#include "src/pke/include/scheme/ckksrns/gen-cryptocontext-ckksrns-params.h"  // from @openfhe
#include "src/pke/include/scheme/ckksrns/gen-cryptocontext-ckksrns.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/naive_matmul/naive_matmul_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

// TODO (#1147): Add support for tensor<ctxt> to add-client-interface
std::vector<CiphertextT> matmul__encrypt__arg0(CryptoContextT v16,
                                               std::vector<double> v17,
                                               PublicKeyT v18) {
  std::vector<float> v17_cast(std::begin(v17), std::end(v17));
  int32_t n =
      v16->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  // create a 1x16 vector of ciphertexts encrypting each value
  std::vector<CiphertextT> outputs;
  outputs.reserve(16);
  for (auto v17_val : v17_cast) {
    std::vector<double> single(n, v17_val);
    const auto& v19 = v16->MakeCKKSPackedPlaintext(single);
    const auto& v20 = v16->Encrypt(v18, v19);
    outputs.push_back(v20);
  }
  return outputs;
}

std::vector<CiphertextT> matmul__encrypt__arg1(CryptoContextT v16,
                                               std::vector<double> v17,
                                               PublicKeyT v18) {
  std::vector<float> v17_cast(std::begin(v17), std::end(v17));
  int32_t n =
      v16->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  // create a 1x16 vector of ciphertexts encrypting each value
  std::vector<CiphertextT> outputs;
  outputs.reserve(16);
  for (auto v17_val : v17_cast) {
    std::vector<double> single(n, v17_val);
    const auto& v19 = v16->MakeCKKSPackedPlaintext(single);
    const auto& v20 = v16->Encrypt(v18, v19);
    outputs.push_back(v20);
  }
  return outputs;
}

double matmul__decrypt__result0(CryptoContextT v26,
                                std::vector<CiphertextT> v27, PrivateKeyT v28) {
  PlaintextT v29;
  v26->Decrypt(v28, v27[0], &v29);  // just decrypt first element
  double v30 = v29->GetCKKSPackedValue()[0].real();
  return v30;
}

TEST(NaiveMatmulTest, RunTest) {
  CCParams<CryptoContextCKKSRNS> parameters;
  parameters.SetMultiplicativeDepth(1);
  // needs to be large enough to accommodate overflow in the plaintext space
  // pick a 32-bit prime for which (p-1) / 65536 is an integer.
  parameters.SetPlaintextModulus(4295294977);
  CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
  cryptoContext->Enable(PKE);
  cryptoContext->Enable(KEYSWITCH);
  cryptoContext->Enable(LEVELEDSHE);

  KeyPair<DCRTPoly> keyPair;
  keyPair = cryptoContext->KeyGen();
  cryptoContext->EvalMultKeyGen(keyPair.secretKey);

  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;

  std::vector<double> arg0Vals = {1.0, 0, 0, 0, 0, 0, 0, 0,
                                  0,   0, 0, 0, 0, 0, 0, 0};  // input
  std::vector<double> arg1Vals = {0.25, 0, 0, 0, 0, 0, 0, 0,
                                  0,    0, 0, 0, 0, 0, 0, 0};  // bias

  // This select the first element of the matrix (0x5036cb3d = 0.0992247) and
  // adds 0.25
  double expected = 0.3492247;

  auto arg0Encrypted =
      matmul__encrypt__arg0(cryptoContext, arg0Vals, publicKey);
  auto arg1Encrypted =
      matmul__encrypt__arg1(cryptoContext, arg1Vals, publicKey);

  // Insert timing info
  std::clock_t c_start = std::clock();
  auto outputEncrypted = matmul(cryptoContext, arg0Encrypted, arg1Encrypted);
  std::clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
  std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";

  auto actual =
      matmul__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-6);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
