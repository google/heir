// RUN: heir-opt --lwe-add-client-interface %s

// CHECK-NOT: 16384
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z35184372744193_i64 = !mod_arith.int<35184372744193 : i64>
!Z35184373006337_i64 = !mod_arith.int<35184373006337 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>
#alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L2_C2 = #lwe.modulus_chain<elements = <35184372121601 : i64, 35184372744193 : i64, 35184373006337 : i64>, current = 2>
!rns_L2 = !rns.rns<!Z35184372121601_i64, !Z35184372744193_i64, !Z35184373006337_i64>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 8192), alignment = #alignment>
#ring_Z65537_i64_1_x8192 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**8192>>
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>
#ring_rns_L2_1_x8192 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**8192>>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<8192xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x8192, encoding = #full_crt_packing_encoding>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x8192, encryption_type = lsb>
#ciphertext_space_L2_D3 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x8192, encryption_type = lsb, size = 3>
!ct_L2 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<8192xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x8192, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
!ct_L2_D3 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<8192xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x8192, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L2_D3, key = #key, modulus_chain = #modulus_chain_L2_C2>
module attributes {backend.openfhe, bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [35184372121601, 35184372744193, 35184373006337], P = [35184373989377, 35184374874113], plaintextModulus = 65537>, scheme.bgv} {
  func.func @dot_product(%ct: !ct_L2 {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xi16>, layout = <map = (d0) -> (d0 mod 8192), alignment = <in = [8], out = [8192]>>>}, %ct_0: !ct_L2 {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xi16>, layout = <map = (d0) -> (d0 mod 8192), alignment = <in = [8], out = [8192]>>>}) -> (!ct_L2 {tensor_ext.original_type = #original_type}) attributes {mgmt.openfhe_params = #mgmt.openfhe_params<evalAddCount = 8, keySwitchCount = 15>} {
    %c7 = arith.constant 7 : index
    %c1_i16 = arith.constant 1 : i16
    %cst = arith.constant dense<0> : tensor<8192xi16>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %inserted = tensor.insert %c1_i16 into %cst[%c7] : tensor<8192xi16>
    %0 = mgmt.init %inserted : tensor<8192xi16>
    %ct_1 = bgv.mul %ct, %ct_0 : (!ct_L2, !ct_L2) -> !ct_L2_D3
    %ct_2 = bgv.relinearize %ct_1 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct_L2_D3 -> !ct_L2
    %ct_3 = bgv.rotate_cols %ct_2 {offset = 4 : index} : !ct_L2
    %ct_4 = bgv.add %ct_2, %ct_3 : (!ct_L2, !ct_L2) -> !ct_L2
    %ct_5 = bgv.rotate_cols %ct_4 {offset = 2 : index} : !ct_L2
    %ct_6 = bgv.add %ct_4, %ct_5 : (!ct_L2, !ct_L2) -> !ct_L2
    %ct_7 = bgv.rotate_cols %ct_6 {offset = 1 : index} : !ct_L2
    %ct_8 = bgv.add %ct_6, %ct_7 : (!ct_L2, !ct_L2) -> !ct_L2
    %pt = lwe.rlwe_encode %inserted {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x8192} : tensor<8192xi16> -> !pt
    %ct_9 = bgv.mul_plain %pt, %ct_8 : (!pt, !ct_L2) -> !ct_L2
    %ct_10 = bgv.rotate_cols %ct_9 {offset = 7 : index} : !ct_L2
    return %ct_10 : !ct_L2
  }
}
