!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
!rns_L0 = !rns.rns<!Z1095233372161_i64>
#ring_Z65537_i64_1_x1024 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = lsb>
!ct_L0 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L5_C0>
module {
  func.func @require_post_pass_toposort_lut3(%arg0: !scifrbool.bootstrap_key_standard, %arg1: !scifrbool.key_switch_key, %arg2: !scifrbool.server_parameters, %arg3: tensor<8x!ct_L0>) -> !ct_L0 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %extracted = tensor.extract %arg3[%c0] : tensor<8x!ct_L0>
    %extracted_0 = tensor.extract %arg3[%c1] : tensor<8x!ct_L0>
    %ct = scifrbool.and %extracted, %extracted_0 : !ct_L0
    %ct_1 = scifrbool.and %ct, %extracted_0 : !ct_L0
    %ct_2 = scifrbool.and %ct, %ct_1 : !ct_L0
    return %ct_2 : !ct_L0
  }
}
