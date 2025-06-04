// RUN: heir-opt %s --lwe-to-lattigo | FileCheck %s

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z4295294977_i64_ = !mod_arith.int<4295294977 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z4295294977_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z4295294977_i64_, polynomialModulus = <1 + x**1024>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z4295294977_i64_1_x1024_, encoding = #full_crt_packing_encoding>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
!pkey_L1_ = !lwe.new_lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024_>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = i64>, plaintext_space = #plaintext_space>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
!ct_L1_ = !lwe.new_lwe_ciphertext<application_data = <message_type = i64>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
module attributes {scheme.bgv} {
  // CHECK: func @foo__encrypt__arg0
  func.func @foo__encrypt__arg0(%arg0: i64, %pk: !pkey_L1_) -> !ct_L1_ {
    // CHECK: tensor.splat
    // CHECK-NEXT: lattigo.bgv.new_plaintext
    // CHECK-NEXT: lattigo.bgv.encode
    %packed = tensor.splat %arg0 : tensor<1024xi64>
    %pt = lwe.rlwe_encode %packed {encoding = #full_crt_packing_encoding, ring = #ring_Z4295294977_i64_1_x1024_} : tensor<1024xi64> -> !pt
    // CHECK-NEXT: lattigo.rlwe.encrypt
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L1_) -> !ct_L1_
    return %ct : !ct_L1_
  }
}
