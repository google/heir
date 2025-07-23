// RUN: heir-opt %s 2>&1 | FileCheck %s

// This simply tests for syntax.

#preserve_overflow = #lwe.preserve_overflow<>
#poly = #polynomial.int_polynomial<x>

#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i4, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!ciphertext = !lwe.new_lwe_ciphertext<application_data = <message_type = i1, overflow = #preserve_overflow>, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: test_valid_lwe_ciphertext
func.func @test_valid_lwe_ciphertext(%arg0 : !ciphertext) -> !ciphertext {
  return %arg0 : !ciphertext
}

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
!ciphertext_rlwe = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: test_valid_rlwe_ciphertext
func.func @test_valid_rlwe_ciphertext(%arg0 : !ciphertext_rlwe) -> !ciphertext_rlwe {
  return %arg0 : !ciphertext_rlwe
}

!secret_key = !lwe.new_lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x1024_>

// CHECK: test_new_lwe_secret_key
func.func @test_new_lwe_secret_key(%arg0 : !secret_key) -> !secret_key {
  return %arg0 :!secret_key
}

!public_key = !lwe.new_lwe_public_key<key = #key, ring = #ring_rns_L0_1_x1024_>

// CHECK: test_new_lwe_public_key
func.func @test_new_lwe_public_key(%arg0 : !public_key) -> !public_key {
  return %arg0 : !public_key
}
