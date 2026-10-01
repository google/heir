#include <bit>
#include <complex>
#include <cstdint>
#include <vector>

// copybara and clang-tidy are weird with the formatting here
// NOLINTBEGIN
#include "gtest/gtest.h"                                   // from @googletest
#include "src/core/include/lattice/stdlatticeparms.h"      // from @openfhe
#include "src/pke/include/constants-defs.h"                // from @openfhe
#include "src/pke/include/gen-cryptocontext.h"             // from @openfhe
#include "src/pke/include/scheme/ckksrns/ckksrns-utils.h"  // from @openfhe
#include "src/pke/include/scheme/ckksrns/gen-cryptocontext-ckksrns.h"  // from @openfhe
// NOLINTEND

using lbcrypto::ADVANCEDSHE;
using lbcrypto::CCParams;
using lbcrypto::CryptoContextCKKSRNS;
using lbcrypto::FIXEDMANUAL;
using lbcrypto::GenCryptoContext;
using lbcrypto::HEStd_NotSet;
using lbcrypto::KEYSWITCH;
using lbcrypto::LEVELEDSHE;
using lbcrypto::PKE;

namespace {

uint32_t ceil_log2(uint32_t x) {
  if (x > 1) {
    return std::bit_width(x - 1);
  }
  return 0;
}

TEST(ChebyshevLevelTest, OpenFHEFormulaMatch) {
  CCParams<CryptoContextCKKSRNS> parameters;
  parameters.SetSecurityLevel(HEStd_NotSet);
  parameters.SetScalingTechnique(FIXEDMANUAL);
  parameters.SetRingDim(8192);
  parameters.SetScalingModSize(50);
  parameters.SetFirstModSize(60);
  parameters.SetMultiplicativeDepth(10);

  auto cc = GenCryptoContext(parameters);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  cc->Enable(ADVANCEDSHE);

  auto keyPair = cc->KeyGen();
  cc->EvalMultKeyGen(keyPair.secretKey);

  std::vector<std::complex<double>> input(8, 1.0);
  auto plaintext = cc->MakeCKKSPackedPlaintext(input);
  auto initial_ciphertext = cc->Encrypt(keyPair.publicKey, plaintext);

  for (int d = 1; d <= 40; ++d) {
    std::vector<double> coefficients(d + 1, 1.0);
    auto result =
        cc->EvalChebyshevSeries(initial_ciphertext, coefficients, -1.0, 1.0);

    int actual = result->GetLevel() - initial_ciphertext->GetLevel();

    int expected = 0;
    if (d < 5) {
      expected = ceil_log2(d) + 1;
    } else {
      auto degs = lbcrypto::ComputeDegreesPS(d);
      uint32_t k = degs[0];
      uint32_t m = degs[1];
      expected = ceil_log2(k) + m;
    }

    EXPECT_EQ(actual, expected) << "Failure at degree d = " << d;
  }
}

}  // namespace
