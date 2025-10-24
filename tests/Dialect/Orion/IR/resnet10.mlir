// RUN: heir-opt %s

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
#inverse_canonical_encoding2 = #lwe.inverse_canonical_encoding<scaling_factor = 160>
#key = #lwe.key<>
#modulus_chain_L10_C10 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 10>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L10 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64, !Z1099503370241_i64, !Z1099502714881_i64>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>>
#ring_rns_L10_1_x65536 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix>
#ciphertext_space_L10_D3 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix, size = 3>
!ct_L10 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
!ct_L10_1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
!ct_L10_2 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
!ct_L10_D3 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L10_D3, key = #key, modulus_chain = #modulus_chain_L10_C10>
!ct_L10_D3_1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32768xf64>>, plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L10_D3, key = #key, modulus_chain = #modulus_chain_L10_C10>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797019488257, 1099512938497, 1099510054913, 1099507695617, 1099515691009, 1099516870657, 1099506515969, 1099504549889, 1099503894529, 1099503370241, 1099502714881], P = [1099511627776], logDefaultScale = 40>} {
  func.func @resnet10(%ct: !ct_L10, %arg0: tensor<198x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg1: tensor<288x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.0.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg2: tensor<288x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.0.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg3: tensor<256x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg4: tensor<784x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg5: tensor<64x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.1.0.shortcut.0", orion.layer_role = "weights", orion.level = 2 : i64}, %arg6: tensor<648x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.conv1", orion.layer_role = "weights", orion.level = 2 : i64}, %arg7: tensor<1800x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.conv2", orion.layer_role = "weights", orion.level = 2 : i64}, %arg8: tensor<200x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "layers.2.0.shortcut.0", orion.layer_role = "weights", orion.level = 2 : i64}, %arg9: tensor<89x32768xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "linear", orion.layer_role = "weights", orion.level = 1 : i64}, %arg10: tensor<32768xf64> {orion.layer_name = "linear", orion.layer_role = "bias", orion.level = 1 : i64}) -> !ct_L10 {
    %ct_0 = orion.linear_transform %ct, %arg0 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 198 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<198x32768xf64>) -> !ct_L10
    %ct_1 = ckks.bootstrap %ct_0 : !ct_L10 -> !ct_L10
    %ct_2 = ckks.bootstrap %ct_1 : !ct_L10 -> !ct_L10
    %ct_3 = orion.chebyshev %ct_2 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_4 = ckks.bootstrap %ct_3 : !ct_L10 -> !ct_L10
    %ct_5 = orion.chebyshev %ct_4 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_6 = ckks.bootstrap %ct_5 : !ct_L10 -> !ct_L10
    %ct_7 = ckks.bootstrap %ct_6 : !ct_L10 -> !ct_L10
    %ct_8 = orion.chebyshev %ct_7 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_9 = ckks.mul %ct_8, %ct_8 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_10 = ckks.relinearize %ct_9 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_11 = ckks.mul %ct_10, %ct_10 : (!ct_L10_1, !ct_L10_1) -> !ct_L10_D3_1
    %ct_12 = ckks.relinearize %ct_11 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3_1) -> !ct_L10_2
    %ct_13 = orion.linear_transform %ct_12, %arg1 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 288 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10_2, tensor<288x32768xf64>) -> !ct_L10_2
    %ct_14 = ckks.bootstrap %ct_13 : !ct_L10_2 -> !ct_L10
    %ct_15 = ckks.bootstrap %ct_14 : !ct_L10 -> !ct_L10
    %ct_16 = orion.chebyshev %ct_15 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_17 = ckks.bootstrap %ct_16 : !ct_L10 -> !ct_L10
    %ct_18 = orion.chebyshev %ct_17 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_19 = ckks.bootstrap %ct_18 : !ct_L10 -> !ct_L10
    %ct_20 = ckks.bootstrap %ct_19 : !ct_L10 -> !ct_L10
    %ct_21 = orion.chebyshev %ct_20 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_22 = ckks.mul %ct_21, %ct_21 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_23 = ckks.relinearize %ct_22 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_24 = ckks.mul %ct_23, %ct_23 : (!ct_L10_1, !ct_L10_1) -> !ct_L10_D3_1
    %ct_25 = ckks.relinearize %ct_24 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3_1) -> !ct_L10_2
    %ct_26 = orion.linear_transform %ct_25, %arg2 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 288 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10_2, tensor<288x32768xf64>) -> !ct_L10_2
    %ct_27 = ckks.bootstrap %ct_26 : !ct_L10_2 -> !ct_L10
    %ct_28 = orion.chebyshev %ct_27 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_29 = ckks.bootstrap %ct_28 : !ct_L10 -> !ct_L10
    %ct_30 = orion.chebyshev %ct_29 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_31 = ckks.bootstrap %ct_30 : !ct_L10 -> !ct_L10
    %ct_32 = ckks.bootstrap %ct_31 : !ct_L10 -> !ct_L10
    %ct_33 = orion.chebyshev %ct_32 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_34 = ckks.mul %ct_33, %ct_33 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_35 = ckks.relinearize %ct_34 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_36 = ckks.mul %ct_35, %ct_35 : (!ct_L10_1, !ct_L10_1) -> !ct_L10_D3_1
    %ct_37 = ckks.relinearize %ct_36 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3_1) -> !ct_L10_2
    %ct_38 = ckks.bootstrap %ct_37 : !ct_L10_2 -> !ct_L10
    %ct_39 = orion.linear_transform %ct_38, %arg3 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 256 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<256x32768xf64>) -> !ct_L10
    %ct_40 = ckks.rotate %ct_39 {offset = 16384 : i32} : !ct_L10
    %ct_41 = ckks.add %ct_40, %ct_39 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_42 = ckks.bootstrap %ct_41 : !ct_L10 -> !ct_L10
    %ct_43 = ckks.bootstrap %ct_42 : !ct_L10 -> !ct_L10
    %ct_44 = orion.chebyshev %ct_43 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_45 = ckks.bootstrap %ct_44 : !ct_L10 -> !ct_L10
    %ct_46 = orion.chebyshev %ct_45 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_47 = ckks.bootstrap %ct_46 : !ct_L10 -> !ct_L10
    %ct_48 = ckks.bootstrap %ct_47 : !ct_L10 -> !ct_L10
    %ct_49 = orion.chebyshev %ct_48 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_50 = ckks.mul %ct_49, %ct_49 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_51 = ckks.relinearize %ct_50 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_52 = ckks.mul %ct_51, %ct_51 : (!ct_L10_1, !ct_L10_1) -> !ct_L10_D3_1
    %ct_53 = ckks.relinearize %ct_52 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3_1) -> !ct_L10_2
    %ct_54 = orion.linear_transform %ct_53, %arg4 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 784 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10_2, tensor<784x32768xf64>) -> !ct_L10_2
    %ct_55 = ckks.rotate %ct_54 {offset = 16384 : i32} : !ct_L10_2
    %ct_56 = ckks.add %ct_55, %ct_54 : (!ct_L10_2, !ct_L10_2) -> !ct_L10_2
    %ct_57 = ckks.bootstrap %ct_56 : !ct_L10_2 -> !ct_L10
    %ct_58 = orion.chebyshev %ct_57 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_59 = ckks.bootstrap %ct_58 : !ct_L10 -> !ct_L10
    %ct_60 = orion.chebyshev %ct_59 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_61 = ckks.bootstrap %ct_60 : !ct_L10 -> !ct_L10
    %ct_62 = ckks.bootstrap %ct_61 : !ct_L10 -> !ct_L10
    %ct_63 = orion.chebyshev %ct_62 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_64 = ckks.mul %ct_63, %ct_63 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_65 = ckks.relinearize %ct_64 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_66 = ckks.mul %ct_65, %ct_65 : (!ct_L10_1, !ct_L10_1) -> !ct_L10_D3_1
    %ct_67 = ckks.relinearize %ct_66 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3_1) -> !ct_L10_2
    %ct_68 = ckks.bootstrap %ct_67 : !ct_L10_2 -> !ct_L10
    %ct_69 = orion.linear_transform %ct_68, %arg5 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 64 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<64x32768xf64>) -> !ct_L10
    %ct_70 = ckks.rotate %ct_69 {offset = 16384 : i32} : !ct_L10
    %ct_71 = ckks.add %ct_70, %ct_69 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_72 = orion.linear_transform %ct_71, %arg6 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 648 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<648x32768xf64>) -> !ct_L10
    %ct_73 = ckks.rotate %ct_72 {offset = 16384 : i32} : !ct_L10
    %ct_74 = ckks.add %ct_73, %ct_72 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_75 = ckks.rotate %ct_74 {offset = 8192 : i32} : !ct_L10
    %ct_76 = ckks.add %ct_75, %ct_74 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_77 = ckks.bootstrap %ct_76 : !ct_L10 -> !ct_L10
    %ct_78 = ckks.bootstrap %ct_77 : !ct_L10 -> !ct_L10
    %ct_79 = orion.chebyshev %ct_78 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_80 = ckks.bootstrap %ct_79 : !ct_L10 -> !ct_L10
    %ct_81 = orion.chebyshev %ct_80 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_82 = ckks.bootstrap %ct_81 : !ct_L10 -> !ct_L10
    %ct_83 = ckks.bootstrap %ct_82 : !ct_L10 -> !ct_L10
    %ct_84 = orion.chebyshev %ct_83 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_85 = ckks.mul %ct_84, %ct_84 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_86 = ckks.relinearize %ct_85 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_87 = ckks.mul %ct_86, %ct_86 : (!ct_L10_1, !ct_L10_1) -> !ct_L10_D3_1
    %ct_88 = ckks.relinearize %ct_87 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3_1) -> !ct_L10_2
    %ct_89 = orion.linear_transform %ct_88, %arg7 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 1800 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10_2, tensor<1800x32768xf64>) -> !ct_L10_2
    %ct_90 = ckks.rotate %ct_89 {offset = 16384 : i32} : !ct_L10_2
    %ct_91 = ckks.add %ct_90, %ct_89 : (!ct_L10_2, !ct_L10_2) -> !ct_L10_2
    %ct_92 = ckks.rotate %ct_91 {offset = 8192 : i32} : !ct_L10_2
    %ct_93 = ckks.add %ct_92, %ct_91 : (!ct_L10_2, !ct_L10_2) -> !ct_L10_2
    %ct_94 = ckks.bootstrap %ct_93 : !ct_L10_2 -> !ct_L10
    %ct_95 = orion.chebyshev %ct_94 {coefficients = [-1.5437065068621281E-22, 0.75601828098297119, 7.4115209650896493E-24, -0.25303265452384949, 3.9355922200500533E-22, 0.15315210819244385, -7.7902292620869194E-23, -0.1109011098742485, -1.132215454117716E-22, 0.087929151952266693, 1.029174679336019E-23, -0.073912657797336578, -7.1981165775152697E-23, 0.064969979226589203, 6.2396644616236723E-23, -0.43697935342788696], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_96 = ckks.bootstrap %ct_95 : !ct_L10 -> !ct_L10
    %ct_97 = orion.chebyshev %ct_96 {coefficients = [6.650302534848487E-24, 1.2368911504745483, 5.1413308989295608E-22, -0.39808535575866699, -3.5209036300668546E-23, 0.22248817980289459, -8.585121482244889E-23, -0.14235951006412506, -1.2667532202571353E-22, 0.095177434384822845, -1.573268091097763E-22, -0.063848823308944702, -2.5584349499902338E-22, 0.04180486872792244, 2.6446414172772963E-22, -0.0401601642370224], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_98 = ckks.bootstrap %ct_97 : !ct_L10 -> !ct_L10
    %ct_99 = ckks.bootstrap %ct_98 : !ct_L10 -> !ct_L10
    %ct_100 = orion.chebyshev %ct_99 {coefficients = [0.50002384185791016, 0.62591469287872314, -4.4641295971814543E-5, -0.18211916089057922, 3.6645389627665281E-5, 0.08313615620136261, -2.6313986381865107E-5, -0.039259493350982666, 1.64710872923024E-5, 0.017457883805036545, -8.9406885308562777E-6, -0.0070134112611413002, 4.1781204345170408E-6, 0.0024784815032035112, -1.6643878097966081E-6, -7.5305596692487597E-4, 5.5763547379683587E-7, 1.9199552480131388E-4, -1.5425030142068863E-7, -3.9858889067545533E-5, 3.4315071673063358E-8, 6.4629844018782023E-6, -5.9043383515700043E-9, -7.6716128205589484E-7, 7.3785672016768444E-10, 5.9265129692676055E-8, -5.9621162173950637E-11, -2.2356401174761231E-9], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    %ct_101 = ckks.mul %ct_100, %ct_100 : (!ct_L10, !ct_L10) -> !ct_L10_D3
    %ct_102 = ckks.relinearize %ct_101 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3) -> !ct_L10_1
    %ct_103 = ckks.mul %ct_102, %ct_102 : (!ct_L10_1, !ct_L10_1) -> !ct_L10_D3_1
    %ct_104 = ckks.relinearize %ct_103 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L10_D3_1) -> !ct_L10_2
    %ct_105 = ckks.bootstrap %ct_104 : !ct_L10_2 -> !ct_L10
    %ct_106 = orion.linear_transform %ct_105, %arg8 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 200 : i32, orion_level = 2 : i32, slots = 32768 : i32} : (!ct_L10, tensor<200x32768xf64>) -> !ct_L10
    %ct_107 = ckks.rotate %ct_106 {offset = 16384 : i32} : !ct_L10
    %ct_108 = ckks.add %ct_107, %ct_106 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_109 = ckks.rotate %ct_108 {offset = 8192 : i32} : !ct_L10
    %ct_110 = ckks.add %ct_109, %ct_108 : (!ct_L10, !ct_L10) -> !ct_L10
    %ct_111 = orion.linear_transform %ct_110, %arg9 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 89 : i32, orion_level = 1 : i32, slots = 32768 : i32} : (!ct_L10, tensor<89x32768xf64>) -> !ct_L10
    %pt = lwe.rlwe_encode %arg10 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x65536} : tensor<32768xf64> -> !pt
    %ct_112 = ckks.add_plain %ct_111, %pt : (!ct_L10, !pt) -> !ct_L10
    return %ct_112 : !ct_L10
  }
}
