!Z1056763241666817029_i64 = !mod_arith.int<1056763241666817029 : i64>
!Z1106058412451299513_i64 = !mod_arith.int<1106058412451299513 : i64>
!Z957769724367225479_i64 = !mod_arith.int<957769724367225479 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 60>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#inverse_canonical_encoding2 = #lwe.inverse_canonical_encoding<scaling_factor = 100>
#key = #lwe.key<>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <1106058412451299513 : i64, 1056763241666817029 : i64>, current = 1>
#modulus_chain_L2_C2 = #lwe.modulus_chain<elements = <1106058412451299513 : i64, 1056763241666817029 : i64, 957769724367225479 : i64>, current = 2>
#ring_f64_1_x131072 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**131072>>
!rns_L1 = !rns.rns<!Z1106058412451299513_i64, !Z1056763241666817029_i64>
!rns_L2 = !rns.rns<!Z1106058412451299513_i64, !Z1056763241666817029_i64, !Z957769724367225479_i64>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding1>>
#ring_rns_L1_1_x131072 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**131072>>
#ring_rns_L2_1_x131072 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**131072>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x131072, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x131072, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
!ct_L2_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 17, Q = [1106058412451299513, 1056763241666817029, 957769724367225479, 919081519653443687, 1030837924888066153, 1084354410096143723, 1135846243351935917, 1087115004561311021, 997960547764032911, 892538949448853293, 1002528331340998513, 1100798419621231379, 981696679688787961, 1061922508412786269], P = [1152921504606846976], logDefaultScale = 60>, scheme.ckks} {
  func.func @in_place(%ct: !ct_L2) -> !ct_L1 {
    %cst = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %ct_0 = ckks.rotate %ct {offset = 0 : i32} : !ct_L2
    %pt = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_1 = ckks.mul_plain %ct_0, %pt : (!ct_L2, !pt) -> !ct_L2_1
    %ct_2 = ckks.rescale %ct_1 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_3 = ckks.rotate %ct {offset = 1 : i32} : !ct_L2
    %ct_4 = ckks.mul_plain %ct_3, %pt : (!ct_L2, !pt) -> !ct_L2_1
    %ct_5 = ckks.rescale %ct_4 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_6 = ckks.add %ct_2, %ct_5 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_7 = ckks.rotate %ct {offset = 2 : i32} : !ct_L2
    %ct_8 = ckks.mul_plain %ct_7, %pt : (!ct_L2, !pt) -> !ct_L2_1
    %ct_9 = ckks.rescale %ct_8 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_10 = ckks.add %ct_6, %ct_9 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_11 = ckks.rotate %ct {offset = 3 : i32} : !ct_L2
    %ct_12 = ckks.mul_plain %ct_11, %pt : (!ct_L2, !pt) -> !ct_L2_1
    %ct_13 = ckks.rescale %ct_12 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_14 = ckks.add %ct_10, %ct_13 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_15 = ckks.rotate %ct {offset = 4 : i32} : !ct_L2
    %ct_16 = ckks.mul_plain %ct_15, %pt : (!ct_L2, !pt) -> !ct_L2_1
    %ct_17 = ckks.rescale %ct_16 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_18 = ckks.add %ct_14, %ct_17 : (!ct_L1, !ct_L1) -> !ct_L1
    return %ct_18 : !ct_L1
  }
}
