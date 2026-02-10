#include <vector>

#include "gmock/gmock.h"              // from @googletest
#include "gtest/gtest.h"              // from @googletest
#include "src/pke/include/openfhe.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/loop_support/loop_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

using namespace lbcrypto;
using ::testing::DoubleNear;
using ::testing::Pointwise;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using PrivateKeyT = PrivateKey<DCRTPoly>;

CryptoContextT override_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(23);
  params.SetKeySwitchTechnique(HYBRID);
  params.SetScalingTechnique(FIXEDMANUAL);
  params.SetScalingModSize(55);
  params.SetFirstModSize(60);
  params.SetRingDim(2048);
  params.SetBatchSize(1024);
  params.SetSecurityLevel(HEStd_NotSet);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  cc->Enable(ADVANCEDSHE);
  cc->Enable(FHE);
  return cc;
}
CryptoContextT override_configure_crypto_context(CryptoContextT cc,
                                                 PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalBootstrapSetup({3, 3});
  cc->EvalBootstrapKeyGen(sk, 1024);
  return cc;
}

TEST(LoopTest, RunTest) {
  // auto cryptoContext = loop__generate_crypto_context();
  auto cryptoContext = override_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  // cryptoContext = loop__configure_crypto_context(cryptoContext, secretKey);
  cryptoContext = override_configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0 = {0.,         0.14285714, 0.28571429, 0.42857143,
                             0.57142857, 0.71428571, 0.85714286, 1.};
  std::vector<float> expected = {-1.,         -1.16666629, -1.39989342,
                                 -1.74687019, -2.29543899, -3.19507837,
                                 -4.66914279, -7.};

  auto arg0Encrypted = loop__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto outputEncrypted = loop(cryptoContext, secretKey, arg0Encrypted);
  auto actual =
      loop__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_THAT(actual, Pointwise(DoubleNear(1e-03), expected));
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
