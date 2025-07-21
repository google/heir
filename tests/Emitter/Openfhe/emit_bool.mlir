// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!Z65537_i64 = !mod_arith.int<65537 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!Z8796093202433_i64 = !mod_arith.int<8796093202433 : i64>
!cc = !openfhe.crypto_context
!params = !openfhe.cc_params
!pk = !openfhe.public_key
!sk = !openfhe.private_key
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <67239937 : i64, 8796093202433 : i64>, current = 0>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <67239937 : i64, 8796093202433 : i64>, current = 1>
!rns_L0 = !rns.rns<!Z67239937_i64>
!rns_L1 = !rns.rns<!Z67239937_i64, !Z8796093202433_i64>
#ring_Z65537_i64_1_x1024 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pt = !lwe.lwe_plaintext<application_data = <message_type = i64>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>>
!pt1 = !lwe.lwe_plaintext<application_data = <message_type = i1>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = lsb>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb>
#ciphertext_space_L1_D3 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb, size = 3>
!ct_L0 = !lwe.lwe_ciphertext<application_data = <message_type = i64>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>
!ct_L1 = !lwe.lwe_ciphertext<application_data = <message_type = i64>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_1 = !lwe.lwe_ciphertext<application_data = <message_type = i1>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_D3 = !lwe.lwe_ciphertext<application_data = <message_type = i64>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1_D3, key = #key, modulus_chain = #modulus_chain_L1_C1>

module attributes {scheme.bgv} {
  // CHECK: CiphertextT emit_bool
  // CHECK-SAME: CryptoContextT [[cc:.*]], bool [[v0:.*]], PublicKeyT [[pk:.*]]) {
  func.func @emit_bool(%cc: !cc, %arg0: i1, %pk: !pk) -> !ct_L1_1 {
    // CHECK: std::vector<bool> [[v1:.*]](1024, [[v0]]);
    // CHECK-NEXT: std::vector<int64_t> [[v2:.*]](std::begin([[v1]]), std::end([[v1]]))
    %splat = tensor.splat %arg0 : tensor<1024xi1>
    %0 = arith.extui %splat : tensor<1024xi1> to tensor<1024xi64>
    %pt = openfhe.make_packed_plaintext %cc, %0 : (!cc, tensor<1024xi64>) -> !pt1
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt1, !pk) -> !ct_L1_1
    return %ct : !ct_L1_1
  }
}
