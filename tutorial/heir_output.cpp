
// #include "openfhe/pke/openfhe.h"  // from @openfhe
#include <openfhe.h>

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using MutableCiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT dot_product(CryptoContextT cc, CiphertextT ct, CiphertextT ct1) {
  std::vector<double> v0 = {0, 0, 0, 0, 0, 0, 0, 1};
  std::vector<double> v1(8, 0.10000000149011612);
  const auto& ct2 = cc->EvalMultNoRelin(ct, ct1);
  const auto& ct3 = cc->Relinearize(ct2);
  auto v1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakeCKKSPackedPlaintext(v1_filled);
  const auto& ct4 = cc->EvalAdd(ct3, pt);
  const auto& ct5 = cc->EvalRotate(ct4, 6);
  const auto& ct6 = cc->EvalRotate(ct3, 7);
  const auto& ct7 = cc->EvalAdd(ct5, ct6);
  const auto& ct8 = cc->EvalAdd(ct7, ct3);
  const auto& ct9 = cc->EvalRotate(ct8, 6);
  const auto& ct10 = cc->EvalAdd(ct9, ct6);
  const auto& ct11 = cc->EvalAdd(ct10, ct3);
  const auto& ct12 = cc->EvalRotate(ct11, 6);
  const auto& ct13 = cc->EvalAdd(ct12, ct6);
  const auto& ct14 = cc->EvalAdd(ct13, ct3);
  const auto& ct15 = cc->EvalRotate(ct14, 7);
  const auto& ct16 = cc->EvalAdd(ct15, ct3);
  auto v0_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v0_filled = v0;
  v0_filled.clear();
  v0_filled.reserve(v0_filled_n);
  for (auto i = 0; i < v0_filled_n; ++i) {
    v0_filled.push_back(v0[i % v0.size()]);
  }
  const auto& pt1 = cc->MakeCKKSPackedPlaintext(v0_filled);
  const auto& ct17 = cc->EvalMult(ct16, pt1);
  const auto& ct18 = cc->EvalRotate(ct17, 7);
  const auto& ct19 = ct18;
  return ct19;
}
CiphertextT dot_product__encrypt__arg0(CryptoContextT cc, std::vector<float> v0,
                                       PublicKeyT pk) {
  std::vector<double> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakeCKKSPackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT dot_product__encrypt__arg1(CryptoContextT cc, std::vector<float> v0,
                                       PublicKeyT pk) {
  std::vector<double> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakeCKKSPackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
float dot_product__decrypt__result0(CryptoContextT cc, CiphertextT ct,
                                    PrivateKeyT sk) {
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  float v0 = pt->GetCKKSPackedValue()[0].real();
  return v0;
}
CryptoContextT dot_product__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(2);
  params.SetKeySwitchTechnique(HYBRID);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT dot_product__configure_crypto_context(CryptoContextT cc,
                                                     PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {6, 7});
  return cc;
}
