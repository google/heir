// RUN: heir-translate %s --emit-openfhe-pke --skip-vector-resizing=false  | FileCheck %s --check-prefix=CHECK --check-prefix=RESIZE
// RUN: heir-translate %s --emit-openfhe-pke --skip-vector-resizing=true | FileCheck %s --check-prefix=CHECK --check-prefix=NO-RESIZE


!Z36028797019389953_i64 = !mod_arith.int<36028797019389953 : i64>
!cc = !openfhe.crypto_context
!params = !openfhe.cc_params
!pk = !openfhe.public_key
!sk = !openfhe.private_key
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L0_C0 = #lwe.modulus_chain<elements = <36028797019389953 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!Z36028797019389953_i64>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L0_C0>
module attributes {scheme.ckks} {

  // CHECK: CiphertextT foo(CryptoContextT cc, CiphertextT ct)
  func.func @foo(%cc: !cc, %ct: !ct_L0) -> !ct_L0 {
    return %ct : !ct_L0
  }

  // CHECK: CiphertextT foo__encrypt__arg0(CryptoContextT cc, std::vector<float> v0, PublicKeyT pk)
  func.func @foo__encrypt__arg0(%cc: !cc, %arg0: tensor<1024xf32>, %pk: !pk) -> !ct_L0 attributes {client.enc_func = {func_name = "foo", index = 0 : i64}} {
    //CHECK: std::vector<double> v1(std::begin(v0), std::end(v0));
    %0 = arith.extf %arg0 : tensor<1024xf32> to tensor<1024xf64>

    //RESIZE: auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
    //RESIZE: auto pt_filled = v1;
    //RESIZE: pt_filled.clear();
    //RESIZE: pt_filled.reserve(pt_filled_n);
    //RESIZE: for (auto i = 0; i < pt_filled_n; ++i) {
    //RESIZE:   pt_filled.push_back(v1[i % v1.size()]);
    //RESIZE: }
    //NO-RESIZE-NOT: auto pt_filled

    //RESIZE: const auto& pt = cc->MakeCKKSPackedPlaintext(pt_filled);
    //NO-RESIZE: const auto& pt = cc->MakeCKKSPackedPlaintext(v1);
    %pt = openfhe.make_ckks_packed_plaintext %cc, %0 : (!cc, tensor<1024xf64>) -> !pt

    //CHECK: const auto& ct = cc->Encrypt(pk, pt
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct_L0

    // CHECK: return ct;
    return %ct : !ct_L0
  }

  // CHECK: std::vector<float> foo__decrypt__result0(CryptoContextT cc, CiphertextT ct, PrivateKeyT sk)
  func.func @foo__decrypt__result0(%cc: !cc, %ct: !ct_L0, %sk: !sk) -> tensor<1024xf32> attributes {client.dec_func = {func_name = "foo", index = 0 : i64}} {
    // CHECK: PlaintextT pt;
    // CHECK: cc->Decrypt(sk, ct, &pt);
    %pt = openfhe.decrypt %cc, %ct, %sk : (!cc, !ct_L0, !sk) -> !pt
    // CHECK: pt->SetLength(1024);
    // CHECK: const auto& v0_cast = pt->GetCKKSPackedValue();
    // CHECK: std::vector<float> v0(v0_cast.size());
    // CHECK: std::transform(std::begin(v0_cast), std::end(v0_cast), std::begin(v0), [](const std::complex<double>& c) { return c.real(); });
    %0 = lwe.rlwe_decode %pt {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : !pt -> tensor<1024xf32>
    // CHECK: return v0;
    return %0 : tensor<1024xf32>
  }

  // CHECK: CryptoContextT foo__generate_crypto_context()
  func.func @foo__generate_crypto_context() -> !cc {
    // CHECK: CCParamsT params;
    // CHECK: params.SetMultiplicativeDepth(1);
    // CHECK: params.SetKeySwitchTechnique(HYBRID);
    %params = openfhe.gen_params  {mulDepth = 1 : i64, plainMod = 0 : i64} : () -> !params
    // CHECK: CryptoContextT cc = GenCryptoContext(params);
    // CHECK: cc->Enable(PKE);
    // CHECK: cc->Enable(KEYSWITCH);
    // CHECK: cc->Enable(LEVELEDSHE);
    %cc = openfhe.gen_context %params {supportFHE = false} : (!params) -> !cc
    // CHECK: return cc;
    return %cc : !cc
  }

  // CHECK: CryptoContextT foo__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk)
  func.func @foo__configure_crypto_context(%cc: !cc, %sk: !sk) -> !cc {
    // CHECK: return cc;
    return %cc : !cc
  }
}
