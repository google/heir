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
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 80>
#key = #lwe.key<>
#modulus_chain_L10_C10 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 10>
#modulus_chain_L8_C8 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64>, current = 8>
#modulus_chain_L9_C9 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64>, current = 9>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L10 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64, !Z1099503370241_i64, !Z1099502714881_i64>
!rns_L8 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64>
!rns_L9 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64, !Z1099503370241_i64>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>>
#ring_rns_L10_1_x65536 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ring_rns_L8_1_x65536 = #polynomial.ring<coefficientType = !rns_L8, polynomialModulus = <1 + x**65536>>
#ring_rns_L9_1_x65536 = #polynomial.ring<coefficientType = !rns_L9, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix>
#ciphertext_space_L10_D3 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix, size = 3>
#ciphertext_space_L8 = #lwe.ciphertext_space<ring = #ring_rns_L8_1_x65536, encryption_type = mix>
#ciphertext_space_L9 = #lwe.ciphertext_space<ring = #ring_rns_L9_1_x65536, encryption_type = mix>
#ciphertext_space_L9_D3 = #lwe.ciphertext_space<ring = #ring_rns_L9_1_x65536, encryption_type = mix, size = 3>
!ct_L10 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
!ct_L10_1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
!ct_L10_D3 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L10_D3, key = #key, modulus_chain = #modulus_chain_L10_C10>
!ct_L8 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L8, key = #key, modulus_chain = #modulus_chain_L8_C8>
!ct_L9 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L9, key = #key, modulus_chain = #modulus_chain_L9_C9>
!ct_L9_1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L9, key = #key, modulus_chain = #modulus_chain_L9_C9>
!ct_L9_D3 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L9_D3, key = #key, modulus_chain = #modulus_chain_L9_C9>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797019488257, 1099512938497, 1099510054913, 1099507695617, 1099515691009, 1099516870657, 1099506515969, 1099504549889, 1099503894529, 1099503370241, 1099502714881], P = [1099511627776], logDefaultScale = 40>} {
  func.func @resnet10(%ct: !ct_L10, %arg0: tensor<198x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg1: tensor<288x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.0.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg2: tensor<288x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.0.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg3: tensor<256x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg4: tensor<784x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg5: tensor<64x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.shortcut.0", orion.layer_role = "weights", orion.level = 2 : i64}, %arg6: tensor<648x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg7: tensor<1800x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg8: tensor<200x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.shortcut.0", orion.layer_role = "weights", orion.level = 2 : i64}, %arg9: tensor<89x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "linear", orion.layer_role = "weights", orion.level = 1 : i64}, %arg10: tensor<32768xf64> {orion.layer_name = "linear", orion.layer_role = "bias", orion.level = 1 : i64}) -> !ct_L10 {
    %ct_0 = orion.linear_transform %ct, %arg0 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 198 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<198x32768xf64>) -> !ct_L10
    %ct_1 = ckks.bootstrap %ct_0 : !ct_L10 -> !ct_L10
    %ct_2 = orion.chebyshev %ct_1 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_3 = orion.chebyshev %ct_2 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_4 = ckks.bootstrap %ct_3 : !ct_L10 -> !ct_L10
    %ct_5 = orion.chebyshev %ct_4 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_6 = ckks.mul %ct_5, %ct_5 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_7 = ckks.relinearize %ct_6 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_8 = ckks.rescale %ct_7 {to_ring = #ring_rns_L9_1_x65536} : !ct_L10_1 -> !ct_L9
    %ct_9 = ckks.mul %ct_8, %ct_8 : (!ct_L9, !ct_L9) -> !ct_L9_D3
    %ct_10 = ckks.relinearize %ct_9 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L9_D3) -> !ct_L9_1
    %ct_11 = ckks.rescale %ct_10 {to_ring = #ring_rns_L8_1_x65536} : !ct_L9_1 -> !ct_L8
    %ct_12 = orion.linear_transform %ct_11, %arg1 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 288 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L8, tensor<288x32768xf64>) -> !ct_L8
    %ct_13 = ckks.bootstrap %ct_12 : !ct_L8 -> !ct_L10
    %ct_14 = orion.chebyshev %ct_13 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_15 = orion.chebyshev %ct_14 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_16 = ckks.bootstrap %ct_15 : !ct_L10 -> !ct_L10
    %ct_17 = orion.chebyshev %ct_16 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_18 = ckks.mul %ct_17, %ct_17 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_19 = ckks.relinearize %ct_18 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_20 = ckks.rescale %ct_19 {to_ring = #ring_rns_L9_1_x65536} : !ct_L10_1 -> !ct_L9
    %ct_21 = ckks.mul %ct_20, %ct_20 : (!ct_L9, !ct_L9) -> !ct_L9_D3
    %ct_22 = ckks.relinearize %ct_21 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L9_D3) -> !ct_L9_1
    %ct_23 = ckks.rescale %ct_22 {to_ring = #ring_rns_L8_1_x65536} : !ct_L9_1 -> !ct_L8
    %ct_24 = orion.linear_transform %ct_23, %arg2 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 288 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L8, tensor<288x32768xf64>) -> !ct_L8
    %ct_25 = orion.chebyshev %ct_24 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L8) -> !ct_L10
    %ct_26 = orion.chebyshev %ct_25 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_27 = ckks.bootstrap %ct_26 : !ct_L10 -> !ct_L10
    %ct_28 = orion.chebyshev %ct_27 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_29 = ckks.mul %ct_28, %ct_28 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_30 = ckks.relinearize %ct_29 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_31 = ckks.rescale %ct_30 {to_ring = #ring_rns_L9_1_x65536} : !ct_L10_1 -> !ct_L9
    %ct_32 = ckks.mul %ct_31, %ct_31 : (!ct_L9, !ct_L9) -> !ct_L9_D3
    %ct_33 = ckks.relinearize %ct_32 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L9_D3) -> !ct_L9_1
    %ct_34 = ckks.rescale %ct_33 {to_ring = #ring_rns_L8_1_x65536} : !ct_L9_1 -> !ct_L8
    %ct_35 = ckks.bootstrap %ct_34 : !ct_L8 -> !ct_L10
    %ct_36 = orion.linear_transform %ct_35, %arg3 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 256 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<256x32768xf64>) -> !ct_L10
    %ct_37 = ckks.rotate %ct_36 {offset = 16384 : i32} : !ct_L10
    %ct_38 = ckks.add %ct_37, %ct_36 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_39 = ckks.bootstrap %ct_38 : !ct_L10 -> !ct_L10
    %ct_40 = orion.chebyshev %ct_39 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_41 = orion.chebyshev %ct_40 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_42 = ckks.bootstrap %ct_41 : !ct_L10 -> !ct_L10
    %ct_43 = orion.chebyshev %ct_42 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_44 = ckks.mul %ct_43, %ct_43 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_45 = ckks.relinearize %ct_44 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_46 = ckks.rescale %ct_45 {to_ring = #ring_rns_L9_1_x65536} : !ct_L10_1 -> !ct_L9
    %ct_47 = ckks.mul %ct_46, %ct_46 : (!ct_L9, !ct_L9) -> !ct_L9_D3
    %ct_48 = ckks.relinearize %ct_47 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L9_D3) -> !ct_L9_1
    %ct_49 = ckks.rescale %ct_48 {to_ring = #ring_rns_L8_1_x65536} : !ct_L9_1 -> !ct_L8
    %ct_50 = orion.linear_transform %ct_49, %arg4 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 784 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L8, tensor<784x32768xf64>) -> !ct_L8
    %ct_51 = ckks.rotate %ct_50 {offset = 16384 : i32} : !ct_L8
    %ct_52 = ckks.add %ct_51, %ct_50 : (!ct_L8, !ct_L8) -> !ct_L8
    %ct_53 = orion.chebyshev %ct_52 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L8) -> !ct_L10
    %ct_54 = orion.chebyshev %ct_53 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_55 = ckks.bootstrap %ct_54 : !ct_L10 -> !ct_L10
    %ct_56 = orion.chebyshev %ct_55 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_57 = ckks.mul %ct_56, %ct_56 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_58 = ckks.relinearize %ct_57 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_59 = ckks.rescale %ct_58 {to_ring = #ring_rns_L9_1_x65536} : !ct_L10_1 -> !ct_L9
    %ct_60 = ckks.mul %ct_59, %ct_59 : (!ct_L9, !ct_L9) -> !ct_L9_D3
    %ct_61 = ckks.relinearize %ct_60 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L9_D3) -> !ct_L9_1
    %ct_62 = ckks.rescale %ct_61 {to_ring = #ring_rns_L8_1_x65536} : !ct_L9_1 -> !ct_L8
    %ct_63 = ckks.bootstrap %ct_62 : !ct_L8 -> !ct_L10
    %ct_64 = orion.linear_transform %ct_63, %arg5 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 64 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<64x32768xf64>) -> !ct_L10
    %ct_65 = ckks.rotate %ct_64 {offset = 16384 : i32} : !ct_L10
    %ct_66 = ckks.add %ct_65, %ct_64 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_67 = orion.linear_transform %ct_66, %arg6 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 648 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<648x32768xf64>) -> !ct_L10
    %ct_68 = ckks.rotate %ct_67 {offset = 16384 : i32} : !ct_L10
    %ct_69 = ckks.add %ct_68, %ct_67 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_70 = ckks.rotate %ct_69 {offset = 8192 : i32} : !ct_L10
    %ct_71 = ckks.add %ct_70, %ct_69 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_72 = ckks.bootstrap %ct_71 : !ct_L10 -> !ct_L10
    %ct_73 = orion.chebyshev %ct_72 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_74 = orion.chebyshev %ct_73 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_75 = ckks.bootstrap %ct_74 : !ct_L10 -> !ct_L10
    %ct_76 = orion.chebyshev %ct_75 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_77 = ckks.mul %ct_76, %ct_76 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_78 = ckks.relinearize %ct_77 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_79 = ckks.rescale %ct_78 {to_ring = #ring_rns_L9_1_x65536} : !ct_L10_1 -> !ct_L9
    %ct_80 = ckks.mul %ct_79, %ct_79 : (!ct_L9, !ct_L9) -> !ct_L9_D3
    %ct_81 = ckks.relinearize %ct_80 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L9_D3) -> !ct_L9_1
    %ct_82 = ckks.rescale %ct_81 {to_ring = #ring_rns_L8_1_x65536} : !ct_L9_1 -> !ct_L8
    %ct_83 = orion.linear_transform %ct_82, %arg7 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 1800 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L8, tensor<1800x32768xf64>) -> !ct_L8
    %ct_84 = ckks.rotate %ct_83 {offset = 16384 : i32} : !ct_L8
    %ct_85 = ckks.add %ct_84, %ct_83 : (!ct_L8, !ct_L8) -> !ct_L8
    %ct_86 = ckks.rotate %ct_85 {offset = 8192 : i32} : !ct_L8
    %ct_87 = ckks.add %ct_86, %ct_85 : (!ct_L8, !ct_L8) -> !ct_L8
    %ct_88 = orion.chebyshev %ct_87 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L8) -> !ct_L10
    %ct_89 = orion.chebyshev %ct_88 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_90 = ckks.bootstrap %ct_89 : !ct_L10 -> !ct_L10
    %ct_91 = orion.chebyshev %ct_90 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_92 = ckks.mul %ct_91, %ct_91 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_93 = ckks.relinearize %ct_92 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_94 = ckks.rescale %ct_93 {to_ring = #ring_rns_L9_1_x65536} : !ct_L10_1 -> !ct_L9
    %ct_95 = ckks.mul %ct_94, %ct_94 : (!ct_L9, !ct_L9) -> !ct_L9_D3
    %ct_96 = ckks.relinearize %ct_95 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L9_D3) -> !ct_L9_1
    %ct_97 = ckks.rescale %ct_96 {to_ring = #ring_rns_L8_1_x65536} : !ct_L9_1 -> !ct_L8
    %ct_98 = ckks.bootstrap %ct_97 : !ct_L8 -> !ct_L10
    %ct_99 = orion.linear_transform %ct_98, %arg8 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 200 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<200x32768xf64>) -> !ct_L10
    %ct_100 = ckks.rotate %ct_99 {offset = 16384 : i32} : !ct_L10
    %ct_101 = ckks.add %ct_100, %ct_99 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_102 = ckks.rotate %ct_101 {offset = 8192 : i32} : !ct_L10
    %ct_103 = ckks.add %ct_102, %ct_101 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_104 = orion.linear_transform %ct_103, %arg9 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 89 : i32, orion_level = 1 : i32, slots = 32768 : i32} : (!ct_L10, tensor<89x32768xf64>) -> !ct_L10
    %pt = lwe.rlwe_encode %arg10 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x65536} : tensor<32768xf64> -> !pt
    %ct_105 = ckks.add_plain %ct_104, %pt : (!ct_L10, !pt) -> !ct_L10
    return %ct_105 : !ct_L10
  }
}
