// RUN: heir-opt --annotate-mgmt %s | FileCheck %s
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>

!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

module {
  // CHECK: func.func @test_lwe
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: {{.*}} {mgmt.mgmt = #mgmt.mgmt<level = 2>}
  func.func @test_lwe(%ct: !ct, %pt: !pt) -> !ct {
    // CHECK: mgmt.modreduce
    %ct1 = mgmt.modreduce %ct : !ct
    // CHECK: mgmt.modreduce
    %ct2 = mgmt.modreduce %ct1 : !ct
    // CHECK: mgmt.init
    // CHECK-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 0>}
    %pt_init = mgmt.init %pt : !pt
    %ct_mul = lwe.rmul_plain %ct2, %pt_init : (!ct, !pt) -> !ct
    return %ct_mul : !ct
  }
}
