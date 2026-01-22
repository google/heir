
#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using ConstCiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

std::vector<float> _assign_layout_15700164067060511435(std::vector<float> v0) {
  [[maybe_unused]] size_t v1 = 0;
  std::vector<float> v2(8, 0);
  [[maybe_unused]] int32_t v3 = 0;
  [[maybe_unused]] int32_t v4 = 1;
  [[maybe_unused]] int32_t v5 = 8;
  std::vector<float> v6 = v2;
  for (auto v7 = 0; v7 < 8; ++v7) {
    size_t v9 = static_cast<size_t>(v7);
    float v10 = v0[v9];
    v6[v9 + 8 * (0)] = v10;
  }
  return v6;
}
std::vector<CiphertextT> loop(CryptoContextT cc, std::vector<CiphertextT> v0) {
  std::vector<double> v1(8, 1);
  [[maybe_unused]] size_t v2 = 0;
  std::vector<float> v3(8, 1);
  const auto& v4 = _assign_layout_15700164067060511435(v3);
  std::vector<float> v5(8);
  for (int64_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
    for (int64_t v5_i1 = 0; v5_i1 < 8; ++v5_i1) {
      v5[v5_i1 + 8 * (v5_i0)] = v4[0 + v5_i1 * 1 + 8 * (0 + v5_i0 * 1)];
    }
  }
  std::vector<double> v6(std::begin(v5), std::end(v5));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v6;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (auto i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v6[i % v6.size()]);
  }
  auto pt = cc->MakeCKKSPackedPlaintext(pt_filled);
  auto ct = v0[0];
  auto ct1 = cc->EvalMult(ct, pt);
  cc->ModReduceInPlace(ct1);
  cc->EvalSubInPlace(ct1, pt);
  std::vector<CiphertextT> v7(1);
  cc->LevelReduceInPlace(ct1, nullptr, 2);
  v7[0] = ct1;
  // std::vector<CiphertextT> v9 = v7;
  for (auto v10 = 0; v10 < 2; ++v10) {
    const auto& ct5 = v7[0];
    const auto& ct6 = cc->EvalBootstrap(ct5);
    auto ct7 = cc->EvalMultNoRelin(ct, ct6);
    cc->RelinearizeInPlace(ct7);
    cc->ModReduceInPlace(ct7);
    cc->EvalSubInPlace(ct7, pt);
    auto pt1_filled_n =
        cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
    auto pt1_filled = v1;
    pt1_filled.clear();
    pt1_filled.reserve(pt1_filled_n);
    for (auto i = 0; i < pt1_filled_n; ++i) {
      pt1_filled.push_back(v1[i % v1.size()]);
    }
    auto pt1 = cc->MakeCKKSPackedPlaintext(pt1_filled);
    auto ct11 = cc->EvalMult(ct, pt1);
    cc->ModReduceInPlace(ct11);
    auto ct13 = cc->EvalMultNoRelin(ct11, ct7);
    cc->RelinearizeInPlace(ct13);
    cc->ModReduceInPlace(ct13);
    cc->EvalSubInPlace(ct13, pt);
    cc->LevelReduceInPlace(ct, nullptr, 1);
    auto ct18 = cc->EvalMult(ct, pt1);
    cc->ModReduceInPlace(ct18);
    auto ct20 = cc->EvalMultNoRelin(ct18, ct13);
    cc->RelinearizeInPlace(ct20);
    cc->ModReduceInPlace(ct20);
    cc->EvalSubInPlace(ct20, pt);
    v7[0] = ct20;
  }
  const auto& ct24 = v7[0];
  const auto& ct25 = cc->EvalBootstrap(ct24);
  auto ct26 = cc->EvalMultNoRelin(ct, ct25);
  cc->RelinearizeInPlace(ct26);
  cc->ModReduceInPlace(ct26);
  cc->EvalSubInPlace(ct26, pt);
  cc->LevelReduceInPlace(ct26, nullptr, 2);
  v7[0] = ct26;
  return v7;
}
std::vector<CiphertextT> loop__encrypt__arg0(CryptoContextT cc,
                                             std::vector<float> v0,
                                             PublicKeyT pk) {
  [[maybe_unused]] size_t v1 = 0;
  std::vector<float> v2(8, 0);
  [[maybe_unused]] int32_t v3 = 0;
  [[maybe_unused]] int32_t v4 = 1;
  [[maybe_unused]] int32_t v5 = 8;
  std::vector<float> v6 = v2;
  for (auto v7 = 0; v7 < 8; ++v7) {
    size_t v9 = static_cast<size_t>(v7);
    float v10 = v0[v9];
    v6[v9 + 8 * (0)] = v10;
  }
  std::vector<float> v12(8);
  for (int64_t v12_i0 = 0; v12_i0 < 1; ++v12_i0) {
    for (int64_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
      v12[v12_i1 + 8 * (v12_i0)] = v6[0 + v12_i1 * 1 + 8 * (0 + v12_i0 * 1)];
    }
  }
  std::vector<double> v13(std::begin(v12), std::end(v12));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v13;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (auto i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v13[i % v13.size()]);
  }
  auto pt = cc->MakeCKKSPackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v14 = {ct};
  return v14;
}
std::vector<float> loop__decrypt__result0(CryptoContextT cc,
                                          std::vector<CiphertextT> v0,
                                          PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 0;
  [[maybe_unused]] int32_t v2 = 8;
  [[maybe_unused]] int32_t v3 = 1;
  [[maybe_unused]] int32_t v4 = 0;
  std::vector<float> v5(8, 0);
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(8);
  const auto& v6_cast = pt->GetCKKSPackedValue();
  std::vector<float> v6(v6_cast.size());
  std::transform(std::begin(v6_cast), std::end(v6_cast), std::begin(v6),
                 [](const std::complex<double>& c) { return c.real(); });
  std::vector<float> v7 = v5;
  for (auto v8 = 0; v8 < 8; ++v8) {
    size_t v10 = static_cast<size_t>(v8);
    float v11 = v6[v10 + 8 * (0)];
    v7[v10] = v11;
  }
  return v7;
}
CryptoContextT loop__generate_crypto_context() {
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
CryptoContextT loop__configure_crypto_context(CryptoContextT cc,
                                              PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalBootstrapSetup({3, 3});
  cc->EvalBootstrapKeyGen(sk, 1024);
  return cc;
}
