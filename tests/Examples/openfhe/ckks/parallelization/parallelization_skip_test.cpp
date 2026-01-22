#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include "gtest/gtest.h"                               // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/ciphertext-fwd.h"            // from @openfhe
#include "src/pke/include/encoding/plaintext-fwd.h"    // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/parallelization/parallelization_skip_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(ParallelizationSkipTest, RunTest) {
  auto cryptoContext = rotations__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  cryptoContext =
      rotations__configure_crypto_context(cryptoContext, keyPair.secretKey);

  std::vector<double> x = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
  size_t encodedLength = x.size();

  lbcrypto::Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(x, 1);
  ptxt->SetLength(encodedLength);
  Ciphertext<lbcrypto::DCRTPoly> encrypted =
      cryptoContext->Encrypt(keyPair.publicKey, ptxt);
  auto encryptedVec = {encrypted};

  auto start = std::chrono::high_resolution_clock::now();
  auto out = rotations(cryptoContext, encryptedVec);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Total time (ms): " << std::setw(20) << duration.count()
            << std::endl;
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
