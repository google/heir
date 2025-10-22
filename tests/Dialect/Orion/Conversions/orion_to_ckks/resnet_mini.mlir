// RUN: heir-opt --orion-to-ckks %s | FileCheck %s

// CHECK: @resnet10
// CHECK-NOT: orion.linear_transform
// CHECK-NOT: orion.chebyshev

!Z1099502714881_i64 = !mod_arith.int<1099502714881 : i64>
!Z1099503370241_i64 = !mod_arith.int<1099503370241 : i64>
!Z1099503894529_i64 = !mod_arith.int<1099503894529 : i64>
!Z1099504549889_i64 = !mod_arith.int<1099504549889 : i64>
!Z1099506515969_i64 = !mod_arith.int<1099506515969 : i64>
!Z1099507695617_i64 = !mod_arith.int<1099507695617 : i64>
!Z1099510054913_i64 = !mod_arith.int<1099510054913 : i64>
!Z1099512938497_i64 = !mod_arith.int<1099512938497 : i64>
!Z1099515691009_i64 = !mod_arith.int<1099515691009 : i64>
!Z1099516870657_i64 = !mod_arith.int<1099516870657 : i64>
!Z36028797019488257_i64 = !mod_arith.int<36028797019488257 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#key = #lwe.key<>
#modulus_chain_L10_C10 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 10>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L10 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64, !Z1099503370241_i64, !Z1099502714881_i64>
#ring_rns_L10_1_x65536 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix>
!ct_L10 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797019488257, 1099512938497, 1099510054913, 1099507695617, 1099515691009, 1099516870657, 1099506515969, 1099504549889, 1099503894529, 1099503370241, 1099502714881], P = [1099511627776], logDefaultScale = 40>} {
  func.func @resnet10(%ct: !ct_L10, %arg0: tensor<198x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg1: tensor<288x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.0.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg2: tensor<288x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.0.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg3: tensor<256x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg4: tensor<784x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg5: tensor<64x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.shortcut.0", orion.layer_role = "weights", orion.level = 2 : i64}, %arg6: tensor<648x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg7: tensor<1800x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg8: tensor<200x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.shortcut.0", orion.layer_role = "weights", orion.level = 2 : i64}, %arg9: tensor<89x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "linear", orion.layer_role = "weights", orion.level = 1 : i64}, %arg10: tensor<32768xf64> {orion.layer_name = "linear", orion.layer_role = "bias", orion.level = 1 : i64}) -> !ct_L10 {
    %ct_0 = orion.linear_transform %ct, %arg0 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 198 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<198x32768xf64>) -> !ct_L10
    %ct_1 = ckks.bootstrap %ct_0 : !ct_L10 -> !ct_L10
    %ct_2 = orion.chebyshev %ct_1 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    return %ct_2 : !ct_L10
  }
}
