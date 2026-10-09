// RUN: heir-opt --pass-pipeline='builtin.module(lwe-annotate-plaintext-level,cse,lwe-annotate-plaintext-level)' --split-input-file --verify-diagnostics %s | FileCheck %s

!Z1005037682689_i64_ = !mod_arith.int<1005037682689 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z998595133441_i64_ = !mod_arith.int<998595133441 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#modulus_chain_L5_C3_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 3>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L3_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_, !Z998595133441_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**32>>
#ring_rns_L3_1_x32_ = #polynomial.ring<coefficientType = !rns_L3_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32_, encryption_type = lsb>
#ciphertext_space_L3_ = #lwe.ciphertext_space<ring = #ring_rns_L3_1_x32_, encryption_type = lsb>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct_L1_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L3_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L3_, key = #key, modulus_chain = #modulus_chain_L5_C3_>

module attributes {backend.lattigo, backend.config_override = {requiresMatchingCiphertextPlaintextLevels = true}} {
  // CHECK: func @different_levels
  func.func @different_levels(%ct1: !ct_L1_, %ct3: !ct_L3_, %value: tensor<32xi64>) -> (!ct_L1_, !ct_L3_) {
    // Two identical encode ops are annotated at their respective use levels,
    // and after annotation they differ so CSE leaves them separate.
    // CHECK: lwe.rlwe_encode
    // CHECK-SAME: level = 1
    // CHECK: lwe.rlwe_encode
    // CHECK-SAME: level = 3
    %pt1 = lwe.rlwe_encode %value <encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_> : tensor<32xi64> -> !pt
    %pt2 = lwe.rlwe_encode %value <encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_> : tensor<32xi64> -> !pt
    %res1 = lwe.rmul_plain %ct1, %pt1 : (!ct_L1_, !pt) -> !ct_L1_
    %res3 = lwe.rmul_plain %ct3, %pt2 : (!ct_L3_, !pt) -> !ct_L3_
    return %res1, %res3 : !ct_L1_, !ct_L3_
  }
}

// -----

!Z1005037682689_i64_ = !mod_arith.int<1005037682689 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z998595133441_i64_ = !mod_arith.int<998595133441 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#modulus_chain_L5_C3_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 3>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L3_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_, !Z998595133441_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**32>>
#ring_rns_L3_1_x32_ = #polynomial.ring<coefficientType = !rns_L3_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32_, encryption_type = lsb>
#ciphertext_space_L3_ = #lwe.ciphertext_space<ring = #ring_rns_L3_1_x32_, encryption_type = lsb>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct_L1_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L3_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L3_, key = #key, modulus_chain = #modulus_chain_L5_C3_>

module attributes {backend.lattigo, backend.config_override = {requiresMatchingCiphertextPlaintextLevels = true}} {
  func.func @shared_encoding(%ct1: !ct_L1_, %ct3: !ct_L3_, %value: tensor<32xi64>) -> (!ct_L1_, !ct_L3_) {
    // expected-error@+1 {{is shared by ciphertext operations at different levels, but the target requires plaintext and ciphertext levels to match}}
    %pt = lwe.rlwe_encode %value <encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_> : tensor<32xi64> -> !pt
    %res1 = lwe.rmul_plain %ct1, %pt : (!ct_L1_, !pt) -> !ct_L1_
    %res3 = lwe.rmul_plain %ct3, %pt : (!ct_L3_, !pt) -> !ct_L3_
    return %res1, %res3 : !ct_L1_, !ct_L3_
  }
}
