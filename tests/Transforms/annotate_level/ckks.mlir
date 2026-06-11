// RUN: heir-opt --annotate-mgmt %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1005037682689_i64_ = !mod_arith.int<1005037682689 : i64>
!Z998595133441_i64_ = !mod_arith.int<998595133441 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L2_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_>
!rns_L3_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_, !Z998595133441_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#ring_rns_L2_1_x1024_ = #polynomial.ring<coefficientType = !rns_L2_, polynomialModulus = <1 + x**1024>>
#ring_rns_L3_1_x1024_ = #polynomial.ring<coefficientType = !rns_L3_, polynomialModulus = <1 + x**1024>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>

!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L2_ = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024_, encryption_type = lsb>
#ciphertext_space_L3_ = #lwe.ciphertext_space<ring = #ring_rns_L3_1_x1024_, encryption_type = lsb>

#modulus_chain_L3_C3_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64>, current = 3>
#modulus_chain_L3_C2_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64>, current = 2>
#modulus_chain_L3_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64>, current = 0>

!ct3 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L3_, key = #key, modulus_chain = #modulus_chain_L3_C3_>
!ct2 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L2_, key = #key, modulus_chain = #modulus_chain_L3_C2_>
!ct0 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L3_C0_>

module {
  // CHECK: func.func @test_ckks
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !{{[a-zA-Z0-9_.]+}} {mgmt.mgmt = #mgmt.mgmt<level = 3>}
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: !{{[a-zA-Z0-9_.]+}}
  func.func @test_ckks(%ct: !ct3, %pt: !pt) -> !ct3 {
    // CHECK: ckks.rescale
    // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 2>
    %ct_reduced = ckks.rescale %ct {to_ring = #ring_rns_L2_1_x1024_} : !ct3 -> !ct2

    // CHECK: ckks.level_reduce
    // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 0>
    %ct_reduced2 = ckks.level_reduce %ct_reduced {levelToDrop = 2 : i64} : !ct2 -> !ct0

    // CHECK: ckks.bootstrap
    // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 3>
    %ct_bootstrapped = ckks.bootstrap %ct_reduced2 {targetLevel = 3 : i64} : !ct0 -> !ct3

    // CHECK: mgmt.init
    // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 3>
    %pt_init = mgmt.init %pt : !pt

    // CHECK: ckks.mul_plain
    // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 3>
    %ct_mul = ckks.mul_plain %ct_bootstrapped, %pt_init : (!ct3, !pt) -> !ct3

    return %ct_mul : !ct3
  }
}
