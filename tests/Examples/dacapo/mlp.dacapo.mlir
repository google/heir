// This was generated from translating the dacapo output using the command hbt
// dacapo 40 MLP SEAL CPU

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
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 80>
#inverse_canonical_encoding2 = #lwe.inverse_canonical_encoding<scaling_factor = 120>
#inverse_canonical_encoding3 = #lwe.inverse_canonical_encoding<scaling_factor = 60>
#inverse_canonical_encoding4 = #lwe.inverse_canonical_encoding<scaling_factor = 100>
#key = #lwe.key<>
#modulus_chain_L13_C13 = #lwe.modulus_chain<elements = <1106058412451299513 : i64, 1056763241666817029 : i64, 957769724367225479 : i64, 919081519653443687 : i64, 1030837924888066153 : i64, 1084354410096143723 : i64, 1135846243351935917 : i64, 1087115004561311021 : i64, 997960547764032911 : i64, 892538949448853293 : i64, 1002528331340998513 : i64, 1100798419621231379 : i64, 981696679688787961 : i64, 1061922508412786269 : i64>, current = 13>
#modulus_chain_L13_C3 = #lwe.modulus_chain<elements = <1106058412451299513 : i64, 1056763241666817029 : i64, 957769724367225479 : i64, 919081519653443687 : i64, 1030837924888066153 : i64, 1084354410096143723 : i64, 1135846243351935917 : i64, 1087115004561311021 : i64, 997960547764032911 : i64, 892538949448853293 : i64, 1002528331340998513 : i64, 1100798419621231379 : i64, 981696679688787961 : i64, 1061922508412786269 : i64>, current = 3>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <1106058412451299513 : i64, 1056763241666817029 : i64>, current = 1>
#modulus_chain_L2_C2 = #lwe.modulus_chain<elements = <1106058412451299513 : i64, 1056763241666817029 : i64, 957769724367225479 : i64>, current = 2>
#ring_f64_1_x131072 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**131072>>
!rns_L1 = !rns.rns<!Z1106058412451299513_i64, !Z1056763241666817029_i64>
!rns_L13 = !rns.rns<!Z1106058412451299513_i64, !Z1056763241666817029_i64, !Z957769724367225479_i64, !Z919081519653443687_i64, !Z1030837924888066153_i64, !Z1084354410096143723_i64, !Z1135846243351935917_i64, !Z1087115004561311021_i64, !Z997960547764032911_i64, !Z892538949448853293_i64, !Z1002528331340998513_i64, !Z1100798419621231379_i64, !Z981696679688787961_i64, !Z1061922508412786269_i64>
!rns_L2 = !rns.rns<!Z1106058412451299513_i64, !Z1056763241666817029_i64, !Z957769724367225479_i64>
!rns_L3 = !rns.rns<!Z1106058412451299513_i64, !Z1056763241666817029_i64, !Z957769724367225479_i64, !Z919081519653443687_i64>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding>>
!pt1 = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding3>>
#ring_rns_L13_1_x131072 = #polynomial.ring<coefficientType = !rns_L13, polynomialModulus = <1 + x**131072>>
#ring_rns_L1_1_x131072 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**131072>>
#ring_rns_L2_1_x131072 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**131072>>
#ring_rns_L3_1_x131072 = #polynomial.ring<coefficientType = !rns_L3, polynomialModulus = <1 + x**131072>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x131072, encryption_type = mix>
#ciphertext_space_L13 = #lwe.ciphertext_space<ring = #ring_rns_L13_1_x131072, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x131072, encryption_type = mix>
#ciphertext_space_L3 = #lwe.ciphertext_space<ring = #ring_rns_L3_1_x131072, encryption_type = mix>
#ciphertext_space_L3_D3 = #lwe.ciphertext_space<ring = #ring_rns_L3_1_x131072, encryption_type = mix, size = 3>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L13 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L13, key = #key, modulus_chain = #modulus_chain_L13_C13>
!ct_L13_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding3>, ciphertext_space = #ciphertext_space_L13, key = #key, modulus_chain = #modulus_chain_L13_C13>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding3>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
!ct_L2_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding4>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
!ct_L3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L3, key = #key, modulus_chain = #modulus_chain_L13_C3>
!ct_L3_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L3, key = #key, modulus_chain = #modulus_chain_L13_C3>
!ct_L3_2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L3, key = #key, modulus_chain = #modulus_chain_L13_C3>
!ct_L3_3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding3>, ciphertext_space = #ciphertext_space_L3, key = #key, modulus_chain = #modulus_chain_L13_C3>
!ct_L3_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x131072, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L3_D3, key = #key, modulus_chain = #modulus_chain_L13_C3>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 17, Q = [1106058412451299513, 1056763241666817029, 957769724367225479, 919081519653443687, 1030837924888066153, 1084354410096143723, 1135846243351935917, 1087115004561311021, 997960547764032911, 892538949448853293, 1002528331340998513, 1100798419621231379, 981696679688787961, 1061922508412786269], P = [1152921504606846976], logDefaultScale = 60>, scheme.ckks} {
  func.func @_hecate_MLP(%ct: !ct_L13) -> !ct_L1 {
    %ct_0 = ckks.level_reduce %ct {levelToDrop = 10 : i64} : !ct_L13 -> !ct_L3
    %ct_1 = ckks.rotate %ct_0 {static_shift = 0 : i32} : !ct_L3
    %cst = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_2 = ckks.mul_plain %ct_1, %pt : (!ct_L3, !pt) -> !ct_L3_1
    %ct_3 = ckks.rotate %ct_0 {static_shift = 1 : i32} : !ct_L3
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_5 = lwe.rlwe_encode %cst_4 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_6 = ckks.mul_plain %ct_3, %pt_5 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_7 = ckks.add %ct_2, %ct_6 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_8 = ckks.rotate %ct_0 {static_shift = 2 : i32} : !ct_L3
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_10 = lwe.rlwe_encode %cst_9 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_11 = ckks.mul_plain %ct_8, %pt_10 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_12 = ckks.add %ct_7, %ct_11 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_13 = ckks.rotate %ct_0 {static_shift = 3 : i32} : !ct_L3
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_15 = lwe.rlwe_encode %cst_14 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_16 = ckks.mul_plain %ct_13, %pt_15 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_17 = ckks.add %ct_12, %ct_16 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_18 = ckks.rotate %ct_0 {static_shift = 4 : i32} : !ct_L3
    %cst_19 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_20 = lwe.rlwe_encode %cst_19 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_21 = ckks.mul_plain %ct_18, %pt_20 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_22 = ckks.add %ct_17, %ct_21 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_23 = ckks.rotate %ct_0 {static_shift = 5 : i32} : !ct_L3
    %cst_24 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_25 = lwe.rlwe_encode %cst_24 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_26 = ckks.mul_plain %ct_23, %pt_25 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_27 = ckks.add %ct_22, %ct_26 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_28 = ckks.rotate %ct_0 {static_shift = 6 : i32} : !ct_L3
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_30 = lwe.rlwe_encode %cst_29 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_31 = ckks.mul_plain %ct_28, %pt_30 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_32 = ckks.add %ct_27, %ct_31 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_33 = ckks.rotate %ct_0 {static_shift = 7 : i32} : !ct_L3
    %cst_34 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_35 = lwe.rlwe_encode %cst_34 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_36 = ckks.mul_plain %ct_33, %pt_35 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_37 = ckks.add %ct_32, %ct_36 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_38 = ckks.rotate %ct_0 {static_shift = 8 : i32} : !ct_L3
    %cst_39 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_40 = lwe.rlwe_encode %cst_39 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_41 = ckks.mul_plain %ct_38, %pt_40 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_42 = ckks.add %ct_37, %ct_41 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_43 = ckks.rotate %ct_0 {static_shift = 9 : i32} : !ct_L3
    %cst_44 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_45 = lwe.rlwe_encode %cst_44 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_46 = ckks.mul_plain %ct_43, %pt_45 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_47 = ckks.add %ct_42, %ct_46 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_48 = ckks.rotate %ct_0 {static_shift = 10 : i32} : !ct_L3
    %cst_49 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_50 = lwe.rlwe_encode %cst_49 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_51 = ckks.mul_plain %ct_48, %pt_50 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_52 = ckks.add %ct_47, %ct_51 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_53 = ckks.rotate %ct_0 {static_shift = 11 : i32} : !ct_L3
    %cst_54 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_55 = lwe.rlwe_encode %cst_54 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_56 = ckks.mul_plain %ct_53, %pt_55 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_57 = ckks.add %ct_52, %ct_56 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_58 = ckks.rotate %ct_0 {static_shift = 12 : i32} : !ct_L3
    %cst_59 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_60 = lwe.rlwe_encode %cst_59 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_61 = ckks.mul_plain %ct_58, %pt_60 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_62 = ckks.add %ct_57, %ct_61 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_63 = ckks.rotate %ct_0 {static_shift = 13 : i32} : !ct_L3
    %cst_64 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_65 = lwe.rlwe_encode %cst_64 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_66 = ckks.mul_plain %ct_63, %pt_65 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_67 = ckks.add %ct_62, %ct_66 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_68 = ckks.rotate %ct_0 {static_shift = 14 : i32} : !ct_L3
    %cst_69 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_70 = lwe.rlwe_encode %cst_69 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_71 = ckks.mul_plain %ct_68, %pt_70 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_72 = ckks.add %ct_67, %ct_71 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_73 = ckks.rotate %ct_0 {static_shift = 15 : i32} : !ct_L3
    %cst_74 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_75 = lwe.rlwe_encode %cst_74 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_76 = ckks.mul_plain %ct_73, %pt_75 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_77 = ckks.add %ct_72, %ct_76 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_78 = ckks.rotate %ct_0 {static_shift = 16 : i32} : !ct_L3
    %cst_79 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_80 = lwe.rlwe_encode %cst_79 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_81 = ckks.mul_plain %ct_78, %pt_80 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_82 = ckks.add %ct_77, %ct_81 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_83 = ckks.rotate %ct_0 {static_shift = 17 : i32} : !ct_L3
    %cst_84 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_85 = lwe.rlwe_encode %cst_84 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_86 = ckks.mul_plain %ct_83, %pt_85 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_87 = ckks.add %ct_82, %ct_86 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_88 = ckks.rotate %ct_0 {static_shift = 18 : i32} : !ct_L3
    %cst_89 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_90 = lwe.rlwe_encode %cst_89 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_91 = ckks.mul_plain %ct_88, %pt_90 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_92 = ckks.add %ct_87, %ct_91 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_93 = ckks.rotate %ct_0 {static_shift = 19 : i32} : !ct_L3
    %cst_94 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_95 = lwe.rlwe_encode %cst_94 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_96 = ckks.mul_plain %ct_93, %pt_95 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_97 = ckks.add %ct_92, %ct_96 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_98 = ckks.rotate %ct_0 {static_shift = 20 : i32} : !ct_L3
    %cst_99 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_100 = lwe.rlwe_encode %cst_99 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_101 = ckks.mul_plain %ct_98, %pt_100 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_102 = ckks.add %ct_97, %ct_101 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_103 = ckks.rotate %ct_0 {static_shift = 21 : i32} : !ct_L3
    %cst_104 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_105 = lwe.rlwe_encode %cst_104 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_106 = ckks.mul_plain %ct_103, %pt_105 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_107 = ckks.add %ct_102, %ct_106 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_108 = ckks.rotate %ct_0 {static_shift = 22 : i32} : !ct_L3
    %cst_109 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_110 = lwe.rlwe_encode %cst_109 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_111 = ckks.mul_plain %ct_108, %pt_110 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_112 = ckks.add %ct_107, %ct_111 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_113 = ckks.rotate %ct_0 {static_shift = 23 : i32} : !ct_L3
    %cst_114 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_115 = lwe.rlwe_encode %cst_114 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_116 = ckks.mul_plain %ct_113, %pt_115 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_117 = ckks.add %ct_112, %ct_116 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_118 = ckks.rotate %ct_0 {static_shift = 24 : i32} : !ct_L3
    %cst_119 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_120 = lwe.rlwe_encode %cst_119 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_121 = ckks.mul_plain %ct_118, %pt_120 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_122 = ckks.add %ct_117, %ct_121 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_123 = ckks.rotate %ct_0 {static_shift = 25 : i32} : !ct_L3
    %cst_124 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_125 = lwe.rlwe_encode %cst_124 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_126 = ckks.mul_plain %ct_123, %pt_125 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_127 = ckks.add %ct_122, %ct_126 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_128 = ckks.rotate %ct_0 {static_shift = 26 : i32} : !ct_L3
    %cst_129 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_130 = lwe.rlwe_encode %cst_129 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_131 = ckks.mul_plain %ct_128, %pt_130 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_132 = ckks.add %ct_127, %ct_131 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_133 = ckks.rotate %ct_0 {static_shift = 27 : i32} : !ct_L3
    %cst_134 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_135 = lwe.rlwe_encode %cst_134 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_136 = ckks.mul_plain %ct_133, %pt_135 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_137 = ckks.add %ct_132, %ct_136 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_138 = ckks.rotate %ct_0 {static_shift = 28 : i32} : !ct_L3
    %cst_139 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_140 = lwe.rlwe_encode %cst_139 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_141 = ckks.mul_plain %ct_138, %pt_140 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_142 = ckks.add %ct_137, %ct_141 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_143 = ckks.rotate %ct_0 {static_shift = 29 : i32} : !ct_L3
    %cst_144 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_145 = lwe.rlwe_encode %cst_144 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_146 = ckks.mul_plain %ct_143, %pt_145 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_147 = ckks.add %ct_142, %ct_146 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_148 = ckks.rotate %ct_0 {static_shift = 30 : i32} : !ct_L3
    %cst_149 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_150 = lwe.rlwe_encode %cst_149 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_151 = ckks.mul_plain %ct_148, %pt_150 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_152 = ckks.add %ct_147, %ct_151 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_153 = ckks.rotate %ct_0 {static_shift = 31 : i32} : !ct_L3
    %cst_154 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_155 = lwe.rlwe_encode %cst_154 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_156 = ckks.mul_plain %ct_153, %pt_155 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_157 = ckks.add %ct_152, %ct_156 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_158 = ckks.rotate %ct_0 {static_shift = 32 : i32} : !ct_L3
    %cst_159 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_160 = lwe.rlwe_encode %cst_159 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_161 = ckks.mul_plain %ct_158, %pt_160 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_162 = ckks.add %ct_157, %ct_161 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_163 = ckks.rotate %ct_0 {static_shift = 33 : i32} : !ct_L3
    %cst_164 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_165 = lwe.rlwe_encode %cst_164 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_166 = ckks.mul_plain %ct_163, %pt_165 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_167 = ckks.add %ct_162, %ct_166 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_168 = ckks.rotate %ct_0 {static_shift = 34 : i32} : !ct_L3
    %cst_169 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_170 = lwe.rlwe_encode %cst_169 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_171 = ckks.mul_plain %ct_168, %pt_170 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_172 = ckks.add %ct_167, %ct_171 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_173 = ckks.rotate %ct_0 {static_shift = 35 : i32} : !ct_L3
    %cst_174 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_175 = lwe.rlwe_encode %cst_174 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_176 = ckks.mul_plain %ct_173, %pt_175 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_177 = ckks.add %ct_172, %ct_176 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_178 = ckks.rotate %ct_0 {static_shift = 36 : i32} : !ct_L3
    %cst_179 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_180 = lwe.rlwe_encode %cst_179 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_181 = ckks.mul_plain %ct_178, %pt_180 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_182 = ckks.add %ct_177, %ct_181 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_183 = ckks.rotate %ct_0 {static_shift = 37 : i32} : !ct_L3
    %cst_184 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_185 = lwe.rlwe_encode %cst_184 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_186 = ckks.mul_plain %ct_183, %pt_185 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_187 = ckks.add %ct_182, %ct_186 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_188 = ckks.rotate %ct_0 {static_shift = 38 : i32} : !ct_L3
    %cst_189 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_190 = lwe.rlwe_encode %cst_189 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_191 = ckks.mul_plain %ct_188, %pt_190 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_192 = ckks.add %ct_187, %ct_191 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_193 = ckks.rotate %ct_0 {static_shift = 39 : i32} : !ct_L3
    %cst_194 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_195 = lwe.rlwe_encode %cst_194 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_196 = ckks.mul_plain %ct_193, %pt_195 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_197 = ckks.add %ct_192, %ct_196 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_198 = ckks.rotate %ct_0 {static_shift = 40 : i32} : !ct_L3
    %cst_199 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_200 = lwe.rlwe_encode %cst_199 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_201 = ckks.mul_plain %ct_198, %pt_200 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_202 = ckks.add %ct_197, %ct_201 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_203 = ckks.rotate %ct_0 {static_shift = 41 : i32} : !ct_L3
    %cst_204 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_205 = lwe.rlwe_encode %cst_204 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_206 = ckks.mul_plain %ct_203, %pt_205 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_207 = ckks.add %ct_202, %ct_206 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_208 = ckks.rotate %ct_0 {static_shift = 42 : i32} : !ct_L3
    %cst_209 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_210 = lwe.rlwe_encode %cst_209 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_211 = ckks.mul_plain %ct_208, %pt_210 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_212 = ckks.add %ct_207, %ct_211 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_213 = ckks.rotate %ct_0 {static_shift = 43 : i32} : !ct_L3
    %cst_214 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_215 = lwe.rlwe_encode %cst_214 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_216 = ckks.mul_plain %ct_213, %pt_215 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_217 = ckks.add %ct_212, %ct_216 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_218 = ckks.rotate %ct_0 {static_shift = 44 : i32} : !ct_L3
    %cst_219 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_220 = lwe.rlwe_encode %cst_219 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_221 = ckks.mul_plain %ct_218, %pt_220 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_222 = ckks.add %ct_217, %ct_221 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_223 = ckks.rotate %ct_0 {static_shift = 45 : i32} : !ct_L3
    %cst_224 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_225 = lwe.rlwe_encode %cst_224 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_226 = ckks.mul_plain %ct_223, %pt_225 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_227 = ckks.add %ct_222, %ct_226 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_228 = ckks.rotate %ct_0 {static_shift = 46 : i32} : !ct_L3
    %cst_229 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_230 = lwe.rlwe_encode %cst_229 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_231 = ckks.mul_plain %ct_228, %pt_230 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_232 = ckks.add %ct_227, %ct_231 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_233 = ckks.rotate %ct_0 {static_shift = 47 : i32} : !ct_L3
    %cst_234 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_235 = lwe.rlwe_encode %cst_234 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_236 = ckks.mul_plain %ct_233, %pt_235 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_237 = ckks.add %ct_232, %ct_236 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_238 = ckks.rotate %ct_0 {static_shift = 48 : i32} : !ct_L3
    %cst_239 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_240 = lwe.rlwe_encode %cst_239 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_241 = ckks.mul_plain %ct_238, %pt_240 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_242 = ckks.add %ct_237, %ct_241 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_243 = ckks.rotate %ct_0 {static_shift = 49 : i32} : !ct_L3
    %cst_244 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_245 = lwe.rlwe_encode %cst_244 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_246 = ckks.mul_plain %ct_243, %pt_245 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_247 = ckks.add %ct_242, %ct_246 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_248 = ckks.rotate %ct_0 {static_shift = 50 : i32} : !ct_L3
    %cst_249 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_250 = lwe.rlwe_encode %cst_249 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_251 = ckks.mul_plain %ct_248, %pt_250 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_252 = ckks.add %ct_247, %ct_251 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_253 = ckks.rotate %ct_0 {static_shift = 51 : i32} : !ct_L3
    %cst_254 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_255 = lwe.rlwe_encode %cst_254 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_256 = ckks.mul_plain %ct_253, %pt_255 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_257 = ckks.add %ct_252, %ct_256 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_258 = ckks.rotate %ct_0 {static_shift = 52 : i32} : !ct_L3
    %cst_259 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_260 = lwe.rlwe_encode %cst_259 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_261 = ckks.mul_plain %ct_258, %pt_260 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_262 = ckks.add %ct_257, %ct_261 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_263 = ckks.rotate %ct_0 {static_shift = 53 : i32} : !ct_L3
    %cst_264 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_265 = lwe.rlwe_encode %cst_264 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_266 = ckks.mul_plain %ct_263, %pt_265 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_267 = ckks.add %ct_262, %ct_266 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_268 = ckks.rotate %ct_0 {static_shift = 54 : i32} : !ct_L3
    %cst_269 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_270 = lwe.rlwe_encode %cst_269 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_271 = ckks.mul_plain %ct_268, %pt_270 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_272 = ckks.add %ct_267, %ct_271 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_273 = ckks.rotate %ct_0 {static_shift = 55 : i32} : !ct_L3
    %cst_274 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_275 = lwe.rlwe_encode %cst_274 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_276 = ckks.mul_plain %ct_273, %pt_275 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_277 = ckks.add %ct_272, %ct_276 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_278 = ckks.rotate %ct_0 {static_shift = 56 : i32} : !ct_L3
    %cst_279 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_280 = lwe.rlwe_encode %cst_279 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_281 = ckks.mul_plain %ct_278, %pt_280 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_282 = ckks.add %ct_277, %ct_281 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_283 = ckks.rotate %ct_0 {static_shift = 57 : i32} : !ct_L3
    %cst_284 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_285 = lwe.rlwe_encode %cst_284 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_286 = ckks.mul_plain %ct_283, %pt_285 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_287 = ckks.add %ct_282, %ct_286 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_288 = ckks.rotate %ct_0 {static_shift = 58 : i32} : !ct_L3
    %cst_289 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_290 = lwe.rlwe_encode %cst_289 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_291 = ckks.mul_plain %ct_288, %pt_290 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_292 = ckks.add %ct_287, %ct_291 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_293 = ckks.rotate %ct_0 {static_shift = 59 : i32} : !ct_L3
    %cst_294 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_295 = lwe.rlwe_encode %cst_294 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_296 = ckks.mul_plain %ct_293, %pt_295 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_297 = ckks.add %ct_292, %ct_296 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_298 = ckks.rotate %ct_0 {static_shift = 60 : i32} : !ct_L3
    %cst_299 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_300 = lwe.rlwe_encode %cst_299 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_301 = ckks.mul_plain %ct_298, %pt_300 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_302 = ckks.add %ct_297, %ct_301 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_303 = ckks.rotate %ct_0 {static_shift = 61 : i32} : !ct_L3
    %cst_304 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_305 = lwe.rlwe_encode %cst_304 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_306 = ckks.mul_plain %ct_303, %pt_305 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_307 = ckks.add %ct_302, %ct_306 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_308 = ckks.rotate %ct_0 {static_shift = 62 : i32} : !ct_L3
    %cst_309 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_310 = lwe.rlwe_encode %cst_309 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_311 = ckks.mul_plain %ct_308, %pt_310 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_312 = ckks.add %ct_307, %ct_311 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_313 = ckks.rotate %ct_0 {static_shift = 63 : i32} : !ct_L3
    %cst_314 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_315 = lwe.rlwe_encode %cst_314 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_316 = ckks.mul_plain %ct_313, %pt_315 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_317 = ckks.add %ct_312, %ct_316 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_318 = ckks.rotate %ct_0 {static_shift = 64 : i32} : !ct_L3
    %cst_319 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_320 = lwe.rlwe_encode %cst_319 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_321 = ckks.mul_plain %ct_318, %pt_320 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_322 = ckks.add %ct_317, %ct_321 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_323 = ckks.rotate %ct_0 {static_shift = 65 : i32} : !ct_L3
    %cst_324 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_325 = lwe.rlwe_encode %cst_324 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_326 = ckks.mul_plain %ct_323, %pt_325 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_327 = ckks.add %ct_322, %ct_326 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_328 = ckks.rotate %ct_0 {static_shift = 66 : i32} : !ct_L3
    %cst_329 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_330 = lwe.rlwe_encode %cst_329 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_331 = ckks.mul_plain %ct_328, %pt_330 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_332 = ckks.add %ct_327, %ct_331 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_333 = ckks.rotate %ct_0 {static_shift = 67 : i32} : !ct_L3
    %cst_334 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_335 = lwe.rlwe_encode %cst_334 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_336 = ckks.mul_plain %ct_333, %pt_335 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_337 = ckks.add %ct_332, %ct_336 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_338 = ckks.rotate %ct_0 {static_shift = 68 : i32} : !ct_L3
    %cst_339 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_340 = lwe.rlwe_encode %cst_339 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_341 = ckks.mul_plain %ct_338, %pt_340 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_342 = ckks.add %ct_337, %ct_341 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_343 = ckks.rotate %ct_0 {static_shift = 69 : i32} : !ct_L3
    %cst_344 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_345 = lwe.rlwe_encode %cst_344 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_346 = ckks.mul_plain %ct_343, %pt_345 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_347 = ckks.add %ct_342, %ct_346 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_348 = ckks.rotate %ct_0 {static_shift = 70 : i32} : !ct_L3
    %cst_349 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_350 = lwe.rlwe_encode %cst_349 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_351 = ckks.mul_plain %ct_348, %pt_350 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_352 = ckks.add %ct_347, %ct_351 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_353 = ckks.rotate %ct_0 {static_shift = 71 : i32} : !ct_L3
    %cst_354 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_355 = lwe.rlwe_encode %cst_354 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_356 = ckks.mul_plain %ct_353, %pt_355 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_357 = ckks.add %ct_352, %ct_356 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_358 = ckks.rotate %ct_0 {static_shift = 72 : i32} : !ct_L3
    %cst_359 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_360 = lwe.rlwe_encode %cst_359 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_361 = ckks.mul_plain %ct_358, %pt_360 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_362 = ckks.add %ct_357, %ct_361 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_363 = ckks.rotate %ct_0 {static_shift = 73 : i32} : !ct_L3
    %cst_364 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_365 = lwe.rlwe_encode %cst_364 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_366 = ckks.mul_plain %ct_363, %pt_365 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_367 = ckks.add %ct_362, %ct_366 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_368 = ckks.rotate %ct_0 {static_shift = 74 : i32} : !ct_L3
    %cst_369 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_370 = lwe.rlwe_encode %cst_369 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_371 = ckks.mul_plain %ct_368, %pt_370 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_372 = ckks.add %ct_367, %ct_371 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_373 = ckks.rotate %ct_0 {static_shift = 75 : i32} : !ct_L3
    %cst_374 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_375 = lwe.rlwe_encode %cst_374 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_376 = ckks.mul_plain %ct_373, %pt_375 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_377 = ckks.add %ct_372, %ct_376 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_378 = ckks.rotate %ct_0 {static_shift = 76 : i32} : !ct_L3
    %cst_379 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_380 = lwe.rlwe_encode %cst_379 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_381 = ckks.mul_plain %ct_378, %pt_380 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_382 = ckks.add %ct_377, %ct_381 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_383 = ckks.rotate %ct_0 {static_shift = 77 : i32} : !ct_L3
    %cst_384 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_385 = lwe.rlwe_encode %cst_384 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_386 = ckks.mul_plain %ct_383, %pt_385 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_387 = ckks.add %ct_382, %ct_386 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_388 = ckks.rotate %ct_0 {static_shift = 78 : i32} : !ct_L3
    %cst_389 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_390 = lwe.rlwe_encode %cst_389 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_391 = ckks.mul_plain %ct_388, %pt_390 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_392 = ckks.add %ct_387, %ct_391 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_393 = ckks.rotate %ct_0 {static_shift = 79 : i32} : !ct_L3
    %cst_394 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_395 = lwe.rlwe_encode %cst_394 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_396 = ckks.mul_plain %ct_393, %pt_395 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_397 = ckks.add %ct_392, %ct_396 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_398 = ckks.rotate %ct_0 {static_shift = 80 : i32} : !ct_L3
    %cst_399 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_400 = lwe.rlwe_encode %cst_399 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_401 = ckks.mul_plain %ct_398, %pt_400 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_402 = ckks.add %ct_397, %ct_401 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_403 = ckks.rotate %ct_0 {static_shift = 81 : i32} : !ct_L3
    %cst_404 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_405 = lwe.rlwe_encode %cst_404 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_406 = ckks.mul_plain %ct_403, %pt_405 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_407 = ckks.add %ct_402, %ct_406 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_408 = ckks.rotate %ct_0 {static_shift = 82 : i32} : !ct_L3
    %cst_409 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_410 = lwe.rlwe_encode %cst_409 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_411 = ckks.mul_plain %ct_408, %pt_410 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_412 = ckks.add %ct_407, %ct_411 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_413 = ckks.rotate %ct_0 {static_shift = 83 : i32} : !ct_L3
    %cst_414 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_415 = lwe.rlwe_encode %cst_414 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_416 = ckks.mul_plain %ct_413, %pt_415 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_417 = ckks.add %ct_412, %ct_416 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_418 = ckks.rotate %ct_0 {static_shift = 84 : i32} : !ct_L3
    %cst_419 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_420 = lwe.rlwe_encode %cst_419 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_421 = ckks.mul_plain %ct_418, %pt_420 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_422 = ckks.add %ct_417, %ct_421 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_423 = ckks.rotate %ct_0 {static_shift = 85 : i32} : !ct_L3
    %cst_424 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_425 = lwe.rlwe_encode %cst_424 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_426 = ckks.mul_plain %ct_423, %pt_425 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_427 = ckks.add %ct_422, %ct_426 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_428 = ckks.rotate %ct_0 {static_shift = 86 : i32} : !ct_L3
    %cst_429 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_430 = lwe.rlwe_encode %cst_429 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_431 = ckks.mul_plain %ct_428, %pt_430 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_432 = ckks.add %ct_427, %ct_431 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_433 = ckks.rotate %ct_0 {static_shift = 87 : i32} : !ct_L3
    %cst_434 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_435 = lwe.rlwe_encode %cst_434 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_436 = ckks.mul_plain %ct_433, %pt_435 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_437 = ckks.add %ct_432, %ct_436 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_438 = ckks.rotate %ct_0 {static_shift = 88 : i32} : !ct_L3
    %cst_439 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_440 = lwe.rlwe_encode %cst_439 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_441 = ckks.mul_plain %ct_438, %pt_440 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_442 = ckks.add %ct_437, %ct_441 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_443 = ckks.rotate %ct_0 {static_shift = 89 : i32} : !ct_L3
    %cst_444 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_445 = lwe.rlwe_encode %cst_444 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_446 = ckks.mul_plain %ct_443, %pt_445 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_447 = ckks.add %ct_442, %ct_446 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_448 = ckks.rotate %ct_0 {static_shift = 90 : i32} : !ct_L3
    %cst_449 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_450 = lwe.rlwe_encode %cst_449 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_451 = ckks.mul_plain %ct_448, %pt_450 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_452 = ckks.add %ct_447, %ct_451 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_453 = ckks.rotate %ct_0 {static_shift = 91 : i32} : !ct_L3
    %cst_454 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_455 = lwe.rlwe_encode %cst_454 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_456 = ckks.mul_plain %ct_453, %pt_455 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_457 = ckks.add %ct_452, %ct_456 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_458 = ckks.rotate %ct_0 {static_shift = 92 : i32} : !ct_L3
    %cst_459 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_460 = lwe.rlwe_encode %cst_459 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_461 = ckks.mul_plain %ct_458, %pt_460 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_462 = ckks.add %ct_457, %ct_461 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_463 = ckks.rotate %ct_0 {static_shift = 93 : i32} : !ct_L3
    %cst_464 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_465 = lwe.rlwe_encode %cst_464 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_466 = ckks.mul_plain %ct_463, %pt_465 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_467 = ckks.add %ct_462, %ct_466 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_468 = ckks.rotate %ct_0 {static_shift = 94 : i32} : !ct_L3
    %cst_469 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_470 = lwe.rlwe_encode %cst_469 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_471 = ckks.mul_plain %ct_468, %pt_470 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_472 = ckks.add %ct_467, %ct_471 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_473 = ckks.rotate %ct_0 {static_shift = 95 : i32} : !ct_L3
    %cst_474 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_475 = lwe.rlwe_encode %cst_474 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_476 = ckks.mul_plain %ct_473, %pt_475 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_477 = ckks.add %ct_472, %ct_476 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_478 = ckks.rotate %ct_0 {static_shift = 96 : i32} : !ct_L3
    %cst_479 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_480 = lwe.rlwe_encode %cst_479 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_481 = ckks.mul_plain %ct_478, %pt_480 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_482 = ckks.add %ct_477, %ct_481 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_483 = ckks.rotate %ct_0 {static_shift = 97 : i32} : !ct_L3
    %cst_484 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_485 = lwe.rlwe_encode %cst_484 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_486 = ckks.mul_plain %ct_483, %pt_485 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_487 = ckks.add %ct_482, %ct_486 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_488 = ckks.rotate %ct_0 {static_shift = 98 : i32} : !ct_L3
    %cst_489 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_490 = lwe.rlwe_encode %cst_489 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_491 = ckks.mul_plain %ct_488, %pt_490 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_492 = ckks.add %ct_487, %ct_491 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_493 = ckks.rotate %ct_0 {static_shift = 99 : i32} : !ct_L3
    %cst_494 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_495 = lwe.rlwe_encode %cst_494 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_496 = ckks.mul_plain %ct_493, %pt_495 : (!ct_L3, !pt) -> !ct_L3_1
    %ct_497 = ckks.add %ct_492, %ct_496 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_498 = ckks.rotate %ct_497 {static_shift = 400 : i32} : !ct_L3_1
    %ct_499 = ckks.add %ct_497, %ct_498 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_500 = ckks.rotate %ct_499 {static_shift = 200 : i32} : !ct_L3_1
    %ct_501 = ckks.add %ct_499, %ct_500 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %ct_502 = ckks.rotate %ct_501 {static_shift = 100 : i32} : !ct_L3_1
    %ct_503 = ckks.add %ct_501, %ct_502 : (!ct_L3_1, !ct_L3_1) -> !ct_L3_1
    %cst_504 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_505 = lwe.rlwe_encode %cst_504 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_506 = ckks.mul_plain %ct_503, %pt_505 : (!ct_L3_1, !pt) -> !ct_L3_2
    %ct_507 = ckks.rescale %ct_506 {to_ring = #ring_rns_L2_1_x131072} : !ct_L3_2 -> !ct_L2
    %ct_508 = ckks.bootstrap %ct_507 : !ct_L2 -> !ct_L13_1
    %ct_509 = ckks.level_reduce %ct_508 {levelToDrop = 10 : i64} : !ct_L13_1 -> !ct_L3_3
    %cst_510 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_511 = lwe.rlwe_encode %cst_510 {encoding = #inverse_canonical_encoding3, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt1
    %ct_512 = ckks.add_plain %ct_509, %pt_511 : (!ct_L3_3, !pt1) -> !ct_L3_3
    %ct_513 = ckks.mul %ct_512, %ct_512 : (!ct_L3_3, !ct_L3_3) -> !ct_L3_D3
    %ct_514 = ckks.relinearize %ct_513 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L3_D3) -> !ct_L3_2
    %ct_515 = ckks.rescale %ct_514 {to_ring = #ring_rns_L2_1_x131072} : !ct_L3_2 -> !ct_L2
    %ct_516 = ckks.rotate %ct_515 {static_shift = 0 : i32} : !ct_L2
    %cst_517 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_518 = lwe.rlwe_encode %cst_517 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_519 = ckks.mul_plain %ct_516, %pt_518 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_520 = ckks.rescale %ct_519 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_521 = ckks.rotate %ct_515 {static_shift = 1 : i32} : !ct_L2
    %cst_522 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_523 = lwe.rlwe_encode %cst_522 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_524 = ckks.mul_plain %ct_521, %pt_523 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_525 = ckks.rescale %ct_524 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_526 = ckks.add %ct_520, %ct_525 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_527 = ckks.rotate %ct_515 {static_shift = 2 : i32} : !ct_L2
    %cst_528 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_529 = lwe.rlwe_encode %cst_528 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_530 = ckks.mul_plain %ct_527, %pt_529 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_531 = ckks.rescale %ct_530 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_532 = ckks.add %ct_526, %ct_531 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_533 = ckks.rotate %ct_515 {static_shift = 3 : i32} : !ct_L2
    %cst_534 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_535 = lwe.rlwe_encode %cst_534 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_536 = ckks.mul_plain %ct_533, %pt_535 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_537 = ckks.rescale %ct_536 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_538 = ckks.add %ct_532, %ct_537 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_539 = ckks.rotate %ct_515 {static_shift = 4 : i32} : !ct_L2
    %cst_540 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_541 = lwe.rlwe_encode %cst_540 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_542 = ckks.mul_plain %ct_539, %pt_541 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_543 = ckks.rescale %ct_542 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_544 = ckks.add %ct_538, %ct_543 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_545 = ckks.rotate %ct_515 {static_shift = 5 : i32} : !ct_L2
    %cst_546 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_547 = lwe.rlwe_encode %cst_546 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_548 = ckks.mul_plain %ct_545, %pt_547 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_549 = ckks.rescale %ct_548 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_550 = ckks.add %ct_544, %ct_549 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_551 = ckks.rotate %ct_515 {static_shift = 6 : i32} : !ct_L2
    %cst_552 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_553 = lwe.rlwe_encode %cst_552 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_554 = ckks.mul_plain %ct_551, %pt_553 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_555 = ckks.rescale %ct_554 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_556 = ckks.add %ct_550, %ct_555 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_557 = ckks.rotate %ct_515 {static_shift = 7 : i32} : !ct_L2
    %cst_558 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_559 = lwe.rlwe_encode %cst_558 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_560 = ckks.mul_plain %ct_557, %pt_559 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_561 = ckks.rescale %ct_560 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_562 = ckks.add %ct_556, %ct_561 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_563 = ckks.rotate %ct_515 {static_shift = 8 : i32} : !ct_L2
    %cst_564 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_565 = lwe.rlwe_encode %cst_564 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_566 = ckks.mul_plain %ct_563, %pt_565 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_567 = ckks.rescale %ct_566 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_568 = ckks.add %ct_562, %ct_567 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_569 = ckks.rotate %ct_515 {static_shift = 9 : i32} : !ct_L2
    %cst_570 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_571 = lwe.rlwe_encode %cst_570 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_572 = ckks.mul_plain %ct_569, %pt_571 : (!ct_L2, !pt) -> !ct_L2_1
    %ct_573 = ckks.rescale %ct_572 {to_ring = #ring_rns_L1_1_x131072} : !ct_L2_1 -> !ct_L1
    %ct_574 = ckks.add %ct_568, %ct_573 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_575 = ckks.rotate %ct_574 {static_shift = 50 : i32} : !ct_L1
    %ct_576 = ckks.add %ct_574, %ct_575 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_577 = ckks.rotate %ct_576 {static_shift = 0 : i32} : !ct_L1
    %ct_578 = ckks.rotate %ct_576 {static_shift = 10 : i32} : !ct_L1
    %ct_579 = ckks.add %ct_577, %ct_578 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_580 = ckks.rotate %ct_576 {static_shift = 20 : i32} : !ct_L1
    %ct_581 = ckks.add %ct_579, %ct_580 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_582 = ckks.rotate %ct_576 {static_shift = 30 : i32} : !ct_L1
    %ct_583 = ckks.add %ct_581, %ct_582 : (!ct_L1, !ct_L1) -> !ct_L1
    %ct_584 = ckks.rotate %ct_576 {static_shift = 40 : i32} : !ct_L1
    %ct_585 = ckks.add %ct_583, %ct_584 : (!ct_L1, !ct_L1) -> !ct_L1
    %cst_586 = arith.constant dense<0.000000e+00> : tensor<65536xf64>
    %pt_587 = lwe.rlwe_encode %cst_586 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x131072} : tensor<65536xf64> -> !pt
    %ct_588 = ckks.add_plain %ct_585, %pt_587 : (!ct_L1, !pt) -> !ct_L1
    return %ct_588 : !ct_L1
  }
}
