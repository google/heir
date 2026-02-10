!Z1002528331340998513_i64 = !mod_arith.int<1002528331340998513 : i64>
!Z1030837924888066153_i64 = !mod_arith.int<1030837924888066153 : i64>
!Z1056763241666817029_i64 = !mod_arith.int<1056763241666817029 : i64>
!Z1061922508412786269_i64 = !mod_arith.int<1061922508412786269 : i64>
!Z1084354410096143723_i64 = !mod_arith.int<1084354410096143723 : i64>
!Z1087115004561311021_i64 = !mod_arith.int<1087115004561311021 : i64>
!Z1100798419621231379_i64 = !mod_arith.int<1100798419621231379 : i64>
!Z1106058412451299513_i64 = !mod_arith.int<1106058412451299513 : i64>
!Z1135846243351935917_i64 = !mod_arith.int<1135846243351935917 : i64>
!Z892538949448853293_i64 = !mod_arith.int<892538949448853293 : i64>
!Z919081519653443687_i64 = !mod_arith.int<919081519653443687 : i64>
!Z957769724367225479_i64 = !mod_arith.int<957769724367225479 : i64>
!Z981696679688787961_i64 = !mod_arith.int<981696679688787961 : i64>
!Z997960547764032911_i64 = !mod_arith.int<997960547764032911 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 60>
#key = #lwe.key<>
#modulus_chain_L13_C13 = #lwe.modulus_chain<elements = <1106058412451299513 : i64, 1056763241666817029 : i64, 957769724367225479 : i64, 919081519653443687 : i64, 1030837924888066153 : i64, 1084354410096143723 : i64, 1135846243351935917 : i64, 1087115004561311021 : i64, 997960547764032911 : i64, 892538949448853293 : i64, 1002528331340998513 : i64, 1100798419621231379 : i64, 981696679688787961 : i64, 1061922508412786269 : i64>, current = 13>
#modulus_chain_L2_C2 = #lwe.modulus_chain<elements = <1106058412451299513 : i64, 1056763241666817029 : i64, 957769724367225479 : i64>, current = 2>
#ring_f64_1_x131072 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**131072>>
!rns_L13 = !rns.rns<!Z1106058412451299513_i64, !Z1056763241666817029_i64, !Z957769724367225479_i64, !Z919081519653443687_i64, !Z1030837924888066153_i64, !Z1084354410096143723_i64, !Z1135846243351935917_i64, !Z1087115004561311021_i64, !Z997960547764032911_i64, !Z892538949448853293_i64, !Z1002528331340998513_i64, !Z1100798419621231379_i64, !Z981696679688787961_i64, !Z1061922508412786269_i64>
!rns_L2 = !rns.rns<!Z1106058412451299513_i64, !Z1056763241666817029_i64, !Z957769724367225479_i64>
#ring_rns_L13_1_x131072 = #polynomial.ring<coefficientType = !rns_L13, polynomialModulus = <1 + x**131072>>
#ring_rns_L2_1_x131072 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**131072>>
#ciphertext_space_L13 = #lwe.ciphertext_space<ring = #ring_rns_L13_1_x131072, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x131072, encryption_type = mix>
!ct_L13 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<65536xf64>>, plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L13, key = #key, modulus_chain = #modulus_chain_L13_C13>
!ct_L2 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<65536xf64>>, plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>

// CHECK-DAG: ![[bEvalType:.*]] = !lattigo.ckks.bootstrapping_evaluator
// CHECK-DAG: ![[evalType:.*]] = !lattigo.ckks.evaluator
// CHECK-DAG: ![[ctType:.*]] = !lattigo.rlwe.ciphertext

// CHECK: func.func @bootstrap
// CHECK-SAME: (%[[bEval:.*]]: ![[bEvalType]],
// CHECK-SAME:  %[[eval:.*]]: ![[evalType]],
// CHECK-SAME: , %[[ct:\w+]]: ![[ctType]])
// CHECK: lattigo.ckks.bootstrap %[[bEval]], %[[ct]]

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [0x200000440001, 0x7fff80001, 0x800280001], P = [0x3ffffffb80001, 0x4000000800001], logDefaultScale = 60>, scheme.ckks} {
  func.func @bootstrap(%ct: !ct_L2) -> !ct_L13 {
    %ct_0 = ckks.bootstrap %ct : !ct_L2 -> !ct_L13
    return %ct_0 : !ct_L13
  }
}
