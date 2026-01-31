#include <iostream>

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

std::vector<float> _assign_layout_6191183397986546506(std::vector<float> v0) {
  [[maybe_unused]] size_t v1 = 0;
  [[maybe_unused]] int32_t v2 = 8;
  std::vector<float> v3(8192, 0);
  [[maybe_unused]] int32_t v4 = 0;
  [[maybe_unused]] int32_t v5 = 1;
  [[maybe_unused]] int32_t v6 = 8192;
  std::vector<float> v7 = v3;
  for (auto v8 = 0; v8 < 8192; ++v8) {
    int32_t v10 = v8 % v2;
    size_t v11 = static_cast<size_t>(v10);
    float v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 8192 * (0)] = v12;
  }
  return v7;
}
std::vector<CiphertextT> loop(CryptoContextT cc, std::vector<CiphertextT> v0) {
  std::vector<double> v1(8192, 1);
  [[maybe_unused]] size_t v2 = 0;
  std::vector<float> v3(8, 1);
  const auto& v4 = _assign_layout_6191183397986546506(v3);
  std::vector<float> v5(8192);
  for (int64_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
    for (int64_t v5_i1 = 0; v5_i1 < 8192; ++v5_i1) {
      v5[v5_i1 + 8192 * (v5_i0)] = v4[0 + v5_i1 * 1 + 8192 * (0 + v5_i0 * 1)];
    }
  }
  std::vector<double> v6(std::begin(v5), std::end(v5));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
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
  std::vector<CiphertextT> v9 = v7;
  for (auto v10 = 0; v10 < 2; ++v10) {
    std::cout << "iter " << v10 << std::endl;
    const auto& ct5 = v9[0];
    const auto& ct6 = cc->EvalBootstrap(ct5);
    std::cout << "bootstrap finished " << std::endl;
    auto ct7 = cc->EvalMultNoRelin(ct, ct6);
    cc->RelinearizeInPlace(ct7);
    cc->ModReduceInPlace(ct7);
    cc->EvalSubInPlace(ct7, pt);
    auto pt1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
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
  std::cout << "Loop finished " << std::endl;
  const auto& ct24 = v9[0];
  const auto& ct25 = cc->EvalBootstrap(ct24);
  auto ct26 = cc->EvalMultNoRelin(ct, ct25);
  cc->RelinearizeInPlace(ct26);
  cc->ModReduceInPlace(ct26);
  cc->EvalSubInPlace(ct26, pt);
  cc->LevelReduceInPlace(ct26, nullptr, 2);
  v7[0] = ct26;
  return v7;
}
std::vector<CiphertextT> loop__encrypt__arg0(CryptoContextT cc, std::vector<float> v0, PublicKeyT pk) {
  [[maybe_unused]] size_t v1 = 0;
  [[maybe_unused]] int32_t v2 = 8;
  std::vector<float> v3(8192, 0);
  [[maybe_unused]] int32_t v4 = 0;
  [[maybe_unused]] int32_t v5 = 1;
  [[maybe_unused]] int32_t v6 = 8192;
  std::vector<float> v7 = v3;
  for (auto v8 = 0; v8 < 8192; ++v8) {
    int32_t v10 = v8 % v2;
    size_t v11 = static_cast<size_t>(v10);
    float v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 8192 * (0)] = v12;
  }
  std::vector<float> v15(8192);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 8192; ++v15_i1) {
      v15[v15_i1 + 8192 * (v15_i0)] = v7[0 + v15_i1 * 1 + 8192 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<double> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (auto i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakeCKKSPackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<float> loop__decrypt__result0(CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 0;
  [[maybe_unused]] int32_t v2 = 8192;
  [[maybe_unused]] int32_t v3 = 8;
  [[maybe_unused]] int32_t v4 = 1;
  [[maybe_unused]] int32_t v5 = 0;
  std::vector<float> v6(8, 0);
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(8192);
  const auto& v7_cast = pt->GetCKKSPackedValue();
  std::vector<float> v7(v7_cast.size());
  std::transform(std::begin(v7_cast), std::end(v7_cast), std::begin(v7), [](const std::complex<double>& c) { return c.real(); });
  std::vector<float> v8 = v6;
  for (auto v9 = 0; v9 < 8192; ++v9) {
    int32_t v11 = v9 % v3;
    size_t v12 = static_cast<size_t>(v9);
    float v13 = v7[v12 + 8192 * (0)];
    size_t v14 = static_cast<size_t>(v11);
    v8[v14] = v13;
  }
  return v8;
}
CryptoContextT loop__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(23);
  params.SetKeySwitchTechnique(HYBRID);
  params.SetScalingModSize(50);
  params.SetFirstModSize(50);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  cc->Enable(ADVANCEDSHE);
  cc->Enable(FHE);
  return cc;
}
CryptoContextT loop__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalBootstrapSetup({3, 3});
  auto numSlots = cc->GetRingDimension() / 2;
  cc->EvalBootstrapKeyGen(sk, numSlots);
  return cc;
}
