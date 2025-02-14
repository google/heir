#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"                               // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/ciphertext-fwd.h"            // from @openfhe
#include "src/pke/include/encoding/plaintext-fwd.h"    // from @openfhe
#include "src/pke/include/openfhe.h"                   // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/simple_ckks_bootstrapping_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(SimpleCKKSBootstrappingTest, RunTest) {
  // WARNING: This test used insecure parameter for faster testing.
  // If you were to use this test as a starter point, please change the
  // insecure flag in BUILD to false.
  auto cryptoContext = simple_ckks_bootstrapping__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  cryptoContext = simple_ckks_bootstrapping__configure_crypto_context(
      cryptoContext, keyPair.secretKey);

  // in accordance with simple_ckks_bootstrapping.mlir function arguments
  uint32_t levelsAvailableAfterBootstrap = 2;

  // 3 for levelBudgetEncode
  // 14 for approxModDepth
  // 3 for levelBudgetDecode
  // extra +1 for FLEXIBLEAUTOEXT
  auto depth = levelsAvailableAfterBootstrap + 3 + 14 + 3 + 1;

  std::vector<double> x = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
  size_t encodedLength = x.size();

  // We start with a depleted ciphertext that has used up all of its levels.
  lbcrypto::Plaintext ptxt =
      cryptoContext->MakeCKKSPackedPlaintext(x, 1, depth - 1);

  auto levelBeforeBootstrapping = depth - ptxt->GetLevel();

  EXPECT_EQ(levelBeforeBootstrapping, 1);

  ptxt->SetLength(encodedLength);

  Ciphertext<lbcrypto::DCRTPoly> encrypted =
      cryptoContext->Encrypt(keyPair.publicKey, ptxt);

  auto out = simple_ckks_bootstrapping(cryptoContext, encrypted);

  auto levelAfterBootstrapping =
      depth - out->GetLevel() - (out->GetNoiseScaleDeg() - 1);

  EXPECT_EQ(levelAfterBootstrapping, levelsAvailableAfterBootstrap);

  lbcrypto::Plaintext result;
  cryptoContext->Decrypt(keyPair.secretKey, out, &result);
  result->SetLength(encodedLength);

  for (size_t i = 0; i < encodedLength; i++) {
    EXPECT_NEAR(x[i], result->GetCKKSPackedValue()[i].real(), 0.01);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
