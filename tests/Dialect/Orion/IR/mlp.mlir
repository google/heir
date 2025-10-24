// RUN: heir-opt %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 26>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 104>
#inverse_canonical_encoding2 = #lwe.inverse_canonical_encoding<scaling_factor = 52>
#key = #lwe.key<>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<4096xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>>
!pt1 = !lwe.lwe_plaintext<application_data = <message_type = tensor<4096xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding2>>
!pt2 = !lwe.lwe_plaintext<application_data = <message_type = tensor<4096xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding1>>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
#ciphertext_space_L5_D3 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix, size = 3>
!ct_L5 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<4096xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>
!ct_L5_1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<4096xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>
!ct_L5_2 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<4096xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>
!ct_L5_D3 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<4096xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L5_D3, key = #key, modulus_chain = #modulus_chain_L5_C5>
!ct_L5_D3_1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<4096xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L5_D3, key = #key, modulus_chain = #modulus_chain_L5_C5>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [67108864], logDefaultScale = 26>} {
  func.func @mlp(%ct: !ct_L5, %arg0: tensor<128x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc1", orion.layer_role = "weights", orion.level = 5 : i64}, %arg1: tensor<4096xf64> {orion.layer_name = "fc1", orion.layer_role = "bias", orion.level = 5 : i64}, %arg2: tensor<128x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc2", orion.layer_role = "weights", orion.level = 3 : i64}, %arg3: tensor<4096xf64> {orion.layer_name = "fc2", orion.layer_role = "bias", orion.level = 3 : i64}, %arg4: tensor<137x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc3", orion.layer_role = "weights", orion.level = 1 : i64}, %arg5: tensor<4096xf64> {orion.layer_name = "fc3", orion.layer_role = "bias", orion.level = 1 : i64}) -> !ct_L5_1 {
    %ct_0 = orion.linear_transform %ct, %arg0 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 128 : i32, orion_level = 5 : i32, slots = 4096 : i32} : (!ct_L5, tensor<128x4096xf64>) -> !ct_L5
    %ct_1 = ckks.rotate %ct_0 {offset = 2048 : i32} : !ct_L5
    %ct_2 = ckks.add %ct_1, %ct_0 : (!ct_L5, !ct_L5) -> !ct_L5
    %ct_3 = ckks.rotate %ct_2 {offset = 1024 : i32} : !ct_L5
    %ct_4 = ckks.add %ct_3, %ct_2 : (!ct_L5, !ct_L5) -> !ct_L5
    %ct_5 = ckks.rotate %ct_4 {offset = 512 : i32} : !ct_L5
    %ct_6 = ckks.add %ct_5, %ct_4 : (!ct_L5, !ct_L5) -> !ct_L5
    %ct_7 = ckks.rotate %ct_6 {offset = 256 : i32} : !ct_L5
    %ct_8 = ckks.add %ct_7, %ct_6 : (!ct_L5, !ct_L5) -> !ct_L5
    %ct_9 = ckks.rotate %ct_8 {offset = 128 : i32} : !ct_L5
    %ct_10 = ckks.add %ct_9, %ct_8 : (!ct_L5, !ct_L5) -> !ct_L5
    %pt = lwe.rlwe_encode %arg1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x8192} : tensor<4096xf64> -> !pt
    %ct_11 = ckks.add_plain %ct_10, %pt : (!ct_L5, !pt) -> !ct_L5
    %ct_12 = ckks.mul %ct_11, %ct_11 : (!ct_L5, !ct_L5) -> !ct_L5_D3
    %ct_13 = ckks.relinearize %ct_12 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L5_D3) -> !ct_L5_2
    %ct_14 = orion.linear_transform %ct_13, %arg2 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 128 : i32, orion_level = 3 : i32, slots = 4096 : i32} : (!ct_L5_2, tensor<128x4096xf64>) -> !ct_L5_2
    %ct_15 = ckks.rotate %ct_14 {offset = 2048 : i32} : !ct_L5_2
    %ct_16 = ckks.add %ct_15, %ct_14 : (!ct_L5_2, !ct_L5_2) -> !ct_L5_2
    %ct_17 = ckks.rotate %ct_16 {offset = 1024 : i32} : !ct_L5_2
    %ct_18 = ckks.add %ct_17, %ct_16 : (!ct_L5_2, !ct_L5_2) -> !ct_L5_2
    %ct_19 = ckks.rotate %ct_18 {offset = 512 : i32} : !ct_L5_2
    %ct_20 = ckks.add %ct_19, %ct_18 : (!ct_L5_2, !ct_L5_2) -> !ct_L5_2
    %ct_21 = ckks.rotate %ct_20 {offset = 256 : i32} : !ct_L5_2
    %ct_22 = ckks.add %ct_21, %ct_20 : (!ct_L5_2, !ct_L5_2) -> !ct_L5_2
    %ct_23 = ckks.rotate %ct_22 {offset = 128 : i32} : !ct_L5_2
    %ct_24 = ckks.add %ct_23, %ct_22 : (!ct_L5_2, !ct_L5_2) -> !ct_L5_2
    %pt_25 = lwe.rlwe_encode %arg3 {encoding = #inverse_canonical_encoding2, ring = #ring_f64_1_x8192} : tensor<4096xf64> -> !pt1
    %ct_26 = ckks.add_plain %ct_24, %pt_25 : (!ct_L5_2, !pt1) -> !ct_L5_2
    %ct_27 = ckks.mul %ct_26, %ct_26 : (!ct_L5_2, !ct_L5_2) -> !ct_L5_D3_1
    %ct_28 = ckks.relinearize %ct_27 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L5_D3_1) -> !ct_L5_1
    %ct_29 = orion.linear_transform %ct_28, %arg4 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 137 : i32, orion_level = 1 : i32, slots = 4096 : i32} : (!ct_L5_1, tensor<137x4096xf64>) -> !ct_L5_1
    %pt_30 = lwe.rlwe_encode %arg5 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1_x8192} : tensor<4096xf64> -> !pt2
    %ct_31 = ckks.add_plain %ct_29, %pt_30 : (!ct_L5_1, !pt2) -> !ct_L5_1
    return %ct_31 : !ct_L5_1
  }
}
