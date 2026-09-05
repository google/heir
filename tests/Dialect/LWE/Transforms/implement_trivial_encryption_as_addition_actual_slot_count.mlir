// RUN: heir-opt --implement-trivial-encryption-as-addition %s | FileCheck %s

// Regression test: the trivially-encrypted zero must be sized by the PHYSICAL
// scheme.actual_slot_count (= ringDim/2 for CKKS), not the logical
// scheme.requested_slot_count. Parameter generation may grow the ring past the
// requested count for security (deep chains), and runtime encodes fill all
// physical slots; a requested-sized zero would be sparse on a full ring and
// mismatch the (actual-sized) encodes and bootstraps emitted by the backend.

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

// requested = 1024 (the user's logical layout request), actual = 8192 (the
// physical slot count of the security-sized ring). The zero constant must use
// 8192.
module attributes {scheme.ckks, scheme.requested_slot_count = 1024 : i64, scheme.actual_slot_count = 8192 : i64} {
  // CHECK: func.func @test_trivial_encrypt_zero
  // CHECK: func @test_trivial_encrypt_zero__encrypt__zero__0(
  // CHECK-NOT: tensor<1024xf64>
  // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<8192xf64>
  // CHECK: lwe.rlwe_encode %[[CST]]
  // CHECK: lwe.rlwe_encrypt
  func.func @test_trivial_encrypt_zero(%pk: !pk_L0, %pt: !pt_L0) -> !ct_L0 {
    %0 = lwe.trivial_encrypt %pt : !pt_L0 -> !ct_L0
    return %0 : !ct_L0
  }
}
