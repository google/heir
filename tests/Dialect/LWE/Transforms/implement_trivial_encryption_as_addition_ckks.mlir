// RUN: heir-opt --implement-trivial-encryption-as-addition %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>
!pk_L0 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L0_1_x1024>
!pt_L0 = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>

module {
  // CHECK: func.func @test_trivial_encrypt_zero
  // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<1024xf64>
  // CHECK: lwe.rlwe_encode %[[CST]]
  func.func @test_trivial_encrypt_zero(%pk: !pk_L0, %pt: !pt_L0) -> !ct_L0 {
    %0 = lwe.trivial_encrypt %pt : !pt_L0 -> !ct_L0
    return %0 : !ct_L0
  }
}
