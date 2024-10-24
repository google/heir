#include <cstdint>
#include <ctime>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"                  // from @googletest
#include "src/pke/include/key/keypair.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/halevi_shoup_matmul_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

CiphertextT matmul__encrypt__arg0(CryptoContextT v16, std::vector<double> v17,
                                  PublicKeyT v18) {
  int32_t n =
      v16->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  std::vector<double> outputs;
  outputs.reserve(n);
  for (int i = 0; i < n; ++i) {
    outputs.push_back(v17[i % 16]);
  }
  const auto& v19 = v16->MakeCKKSPackedPlaintext(outputs);
  const auto& v20 = v16->Encrypt(v18, v19);
  return v20;
}

double matmul__decrypt__result0(CryptoContextT v26, CiphertextT v27,
                                PrivateKeyT v28) {
  PlaintextT v29;
  v26->Decrypt(v28, v27, &v29);
  double v30 = v29->GetCKKSPackedValue()[0].real();
  return v30;
}

TEST(NaiveMatmulTest, RunTest) {
  auto cryptoContext = matmul__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = matmul__configure_crypto_context(cryptoContext, secretKey);

  std::vector<double> arg0Vals = {1.0, 0, 0, 0, 0, 0, 0, 0,
                                  0,   0, 0, 0, 0, 0, 0, 0};  // input

  // This select the first element of the matrix (0x5036cb3d =
  // 0.099224686622619628) and adds -0.45141533017158508
  double expected = -0.35219;

  // TODO(#645): support cyclic repetition in add-client-interface
  // TODO(#891): support other schemes besides BGV in add-client-interface
  auto arg0Encrypted =
      matmul__encrypt__arg0(cryptoContext, arg0Vals, publicKey);

  // Insert timing info
  std::clock_t c_start = std::clock();
  auto outputEncrypted = matmul(cryptoContext, arg0Encrypted);
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
