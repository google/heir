#include <ctime>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"                  // from @googletest
#include "src/pke/include/key/keypair.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/halevi_shoup_matvec/debug_helper.h"
#include "tests/Examples/openfhe/ckks/halevi_shoup_matvec/halevi_shoup_matvec_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(NaiveMatmulTest, RunTest) {
  auto cryptoContext = matvec__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = matvec__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0Vals = {1.0, 0, 0, 0, 0, 0, 0, 0,
                                 0,   0, 0, 0, 0, 0, 0, 0};  // input

  // This select the first element of the matrix (0x5036cb3d =
  // 0.099224686622619628) and adds -0.45141533017158508
  float expected = -0.35219;

  auto arg0Encrypted =
      matvec__encrypt__arg0(cryptoContext, arg0Vals, publicKey);

  // Insert timing info
  std::clock_t cStart = std::clock();
  auto outputEncrypted = matvec(cryptoContext, secretKey, arg0Encrypted);
  std::clock_t cEnd = std::clock();
  double timeElapsedMs = 1000.0 * (cEnd - cStart) / CLOCKS_PER_SEC;
  std::cout << "CPU time used: " << timeElapsedMs << " ms\n";

  auto actual =
      matvec__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual.front(), 1e-6);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
