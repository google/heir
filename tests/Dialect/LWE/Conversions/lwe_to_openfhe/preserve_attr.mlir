// RUN: heir-opt --lwe-to-openfhe %s | FileCheck %s

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
!skey_L1_ = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x1024_>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
!ct_L1_ = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

module attributes {scheme.bgv} {
  func.func private @__heir_debug_0(!skey_L1_, !ct_L1_)
  func.func @test_ops(%sk: !skey_L1_, %ct: !ct_L1_, %ct_0: !ct_L1_) {
    %ct_1 = lwe.radd %ct, %ct_0 : (!ct_L1_, !ct_L1_) -> !ct_L1_
    // CHECK: call
    // CHECK-SAME: {message.size = "1"}
    call @__heir_debug_0(%sk, %ct_1) {message.size = "1"} : (!skey_L1_, !ct_L1_) -> ()
    return
  }
}
