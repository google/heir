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

TEST(NaiveMatmulTest, RunTest) {
  auto cryptoContext = matmul__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = matmul__configure_crypto_context(cryptoContext, secretKey);

  std::vector<double> arg0 = {1.0, 0, 0, 0, 0, 0, 0, 0,
                              0,   0, 0, 0, 0, 0, 0, 0};  // input
  std::vector<double> arg1 = {0.25, 0, 0, 0, 0, 0, 0, 0,
                              0,    0, 0, 0, 0, 0, 0, 0};  // bias

  // This select the first element of the matrix (0x5036cb3d = 0.0992247) and
  // adds 0.25
  double expected = 0.3492247;

  auto arg0Encrypted = matmul__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted = matmul__encrypt__arg0(cryptoContext, arg1, publicKey);

  std::clock_t cStart = std::clock();
  auto outputEncrypted = matmul(cryptoContext, arg0Encrypted, arg1Encrypted);
  std::clock_t cEnd = std::clock();
  double timeElapsedMs = 1000.0 * (cEnd - cStart) / CLOCKS_PER_SEC;
  std::cout << "CPU time used: " << timeElapsedMs << " ms\n";

  auto actual =
      matmul__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-3);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
