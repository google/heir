!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z35184372744193_i64 = !mod_arith.int<35184372744193 : i64>
!Z36028797019389953_i64 = !mod_arith.int<36028797019389953 : i64>
#alignment = #tensor_ext.alignment<in = [16], out = [1024]>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 90>
#inverse_canonical_encoding2 = #lwe.inverse_canonical_encoding<scaling_factor = 135>
#key = #lwe.key<>
#modulus_chain_L2_C0 = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64, 35184372744193 : i64>, current = 0>
#modulus_chain_L2_C1 = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64, 35184372744193 : i64>, current = 1>
#modulus_chain_L2_C2 = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64, 35184372744193 : i64>, current = 2>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!Z36028797019389953_i64>
!rns_L1 = !rns.rns<!Z36028797019389953_i64, !Z35184372121601_i64>
!rns_L2 = !rns.rns<!Z36028797019389953_i64, !Z35184372121601_i64, !Z35184372744193_i64>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
!pt1 = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding1>>
#original_type = #tensor_ext.original_type<originalType = tensor<16xf32>, layout = #layout>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ring_rns_L2_1_x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
!pkey_L2 = !lwe.new_lwe_public_key<key = #key, ring = #ring_rns_L2_1_x1024>
!skey_L0 = !lwe.new_lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x1024>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024, encryption_type = mix>
#ciphertext_space_L2_D3 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024, encryption_type = mix, size = 3>
!ct_L0 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L2_C0>
!ct_L1 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L2_C1>
!ct_L1_1 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L2_C1>
!ct_L2 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
!ct_L2_1 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
!ct_L2_2 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
!ct_L2_D3 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L2_D3, key = #key, modulus_chain = #modulus_chain_L2_C2>
!ct_L2_D3_1 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L2_D3, key = #key, modulus_chain = #modulus_chain_L2_C2>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  func.func @polynomial_evaluation(%ct: !ct_L2 {tensor_ext.original_type = #original_type}, %ct_0: !ct_L2 {tensor_ext.original_type = #original_type}) -> (!ct_L0 {tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32>
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %cst_2 = arith.constant dense<3.000000e+00> : tensor<16xf32>
    %cst_3 = arith.constant dense<2.000000e+00> : tensor<16xf32>
    %0 = tensor.empty() : tensor<1024xf32>
    %inserted_slice = tensor.insert_slice %cst_2 into %0[0] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_4 = tensor.insert_slice %cst_2 into %inserted_slice[16] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_5 = tensor.insert_slice %cst_2 into %inserted_slice_4[32] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_6 = tensor.insert_slice %cst_2 into %inserted_slice_5[48] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_7 = tensor.insert_slice %cst_2 into %inserted_slice_6[64] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_8 = tensor.insert_slice %cst_2 into %inserted_slice_7[80] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_9 = tensor.insert_slice %cst_2 into %inserted_slice_8[96] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_10 = tensor.insert_slice %cst_2 into %inserted_slice_9[112] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_11 = tensor.insert_slice %cst_2 into %inserted_slice_10[128] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_12 = tensor.insert_slice %cst_2 into %inserted_slice_11[144] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_13 = tensor.insert_slice %cst_2 into %inserted_slice_12[160] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_14 = tensor.insert_slice %cst_2 into %inserted_slice_13[176] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_15 = tensor.insert_slice %cst_2 into %inserted_slice_14[192] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_16 = tensor.insert_slice %cst_2 into %inserted_slice_15[208] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_17 = tensor.insert_slice %cst_2 into %inserted_slice_16[224] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_18 = tensor.insert_slice %cst_2 into %inserted_slice_17[240] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_19 = tensor.insert_slice %cst_2 into %inserted_slice_18[256] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_20 = tensor.insert_slice %cst_2 into %inserted_slice_19[272] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_21 = tensor.insert_slice %cst_2 into %inserted_slice_20[288] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_22 = tensor.insert_slice %cst_2 into %inserted_slice_21[304] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_23 = tensor.insert_slice %cst_2 into %inserted_slice_22[320] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_24 = tensor.insert_slice %cst_2 into %inserted_slice_23[336] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_25 = tensor.insert_slice %cst_2 into %inserted_slice_24[352] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_26 = tensor.insert_slice %cst_2 into %inserted_slice_25[368] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_27 = tensor.insert_slice %cst_2 into %inserted_slice_26[384] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_28 = tensor.insert_slice %cst_2 into %inserted_slice_27[400] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_29 = tensor.insert_slice %cst_2 into %inserted_slice_28[416] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_30 = tensor.insert_slice %cst_2 into %inserted_slice_29[432] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_31 = tensor.insert_slice %cst_2 into %inserted_slice_30[448] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_32 = tensor.insert_slice %cst_2 into %inserted_slice_31[464] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_33 = tensor.insert_slice %cst_2 into %inserted_slice_32[480] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_34 = tensor.insert_slice %cst_2 into %inserted_slice_33[496] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_35 = tensor.insert_slice %cst_2 into %inserted_slice_34[512] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_36 = tensor.insert_slice %cst_2 into %inserted_slice_35[528] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_37 = tensor.insert_slice %cst_2 into %inserted_slice_36[544] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_38 = tensor.insert_slice %cst_2 into %inserted_slice_37[560] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_39 = tensor.insert_slice %cst_2 into %inserted_slice_38[576] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_40 = tensor.insert_slice %cst_2 into %inserted_slice_39[592] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_41 = tensor.insert_slice %cst_2 into %inserted_slice_40[608] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_42 = tensor.insert_slice %cst_2 into %inserted_slice_41[624] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_43 = tensor.insert_slice %cst_2 into %inserted_slice_42[640] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_44 = tensor.insert_slice %cst_2 into %inserted_slice_43[656] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_45 = tensor.insert_slice %cst_2 into %inserted_slice_44[672] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_46 = tensor.insert_slice %cst_2 into %inserted_slice_45[688] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_47 = tensor.insert_slice %cst_2 into %inserted_slice_46[704] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_48 = tensor.insert_slice %cst_2 into %inserted_slice_47[720] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_49 = tensor.insert_slice %cst_2 into %inserted_slice_48[736] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_50 = tensor.insert_slice %cst_2 into %inserted_slice_49[752] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_51 = tensor.insert_slice %cst_2 into %inserted_slice_50[768] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_52 = tensor.insert_slice %cst_2 into %inserted_slice_51[784] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_53 = tensor.insert_slice %cst_2 into %inserted_slice_52[800] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_54 = tensor.insert_slice %cst_2 into %inserted_slice_53[816] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_55 = tensor.insert_slice %cst_2 into %inserted_slice_54[832] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_56 = tensor.insert_slice %cst_2 into %inserted_slice_55[848] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_57 = tensor.insert_slice %cst_2 into %inserted_slice_56[864] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_58 = tensor.insert_slice %cst_2 into %inserted_slice_57[880] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_59 = tensor.insert_slice %cst_2 into %inserted_slice_58[896] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_60 = tensor.insert_slice %cst_2 into %inserted_slice_59[912] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_61 = tensor.insert_slice %cst_2 into %inserted_slice_60[928] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_62 = tensor.insert_slice %cst_2 into %inserted_slice_61[944] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_63 = tensor.insert_slice %cst_2 into %inserted_slice_62[960] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_64 = tensor.insert_slice %cst_2 into %inserted_slice_63[976] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_65 = tensor.insert_slice %cst_2 into %inserted_slice_64[992] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_66 = tensor.insert_slice %cst_2 into %inserted_slice_65[1008] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_67 = tensor.insert_slice %cst_3 into %0[0] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_68 = tensor.insert_slice %cst_3 into %inserted_slice_67[16] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_69 = tensor.insert_slice %cst_3 into %inserted_slice_68[32] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_70 = tensor.insert_slice %cst_3 into %inserted_slice_69[48] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_71 = tensor.insert_slice %cst_3 into %inserted_slice_70[64] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_72 = tensor.insert_slice %cst_3 into %inserted_slice_71[80] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_73 = tensor.insert_slice %cst_3 into %inserted_slice_72[96] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_74 = tensor.insert_slice %cst_3 into %inserted_slice_73[112] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_75 = tensor.insert_slice %cst_3 into %inserted_slice_74[128] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_76 = tensor.insert_slice %cst_3 into %inserted_slice_75[144] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_77 = tensor.insert_slice %cst_3 into %inserted_slice_76[160] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_78 = tensor.insert_slice %cst_3 into %inserted_slice_77[176] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_79 = tensor.insert_slice %cst_3 into %inserted_slice_78[192] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_80 = tensor.insert_slice %cst_3 into %inserted_slice_79[208] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_81 = tensor.insert_slice %cst_3 into %inserted_slice_80[224] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_82 = tensor.insert_slice %cst_3 into %inserted_slice_81[240] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_83 = tensor.insert_slice %cst_3 into %inserted_slice_82[256] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_84 = tensor.insert_slice %cst_3 into %inserted_slice_83[272] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_85 = tensor.insert_slice %cst_3 into %inserted_slice_84[288] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_86 = tensor.insert_slice %cst_3 into %inserted_slice_85[304] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_87 = tensor.insert_slice %cst_3 into %inserted_slice_86[320] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_88 = tensor.insert_slice %cst_3 into %inserted_slice_87[336] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_89 = tensor.insert_slice %cst_3 into %inserted_slice_88[352] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_90 = tensor.insert_slice %cst_3 into %inserted_slice_89[368] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_91 = tensor.insert_slice %cst_3 into %inserted_slice_90[384] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_92 = tensor.insert_slice %cst_3 into %inserted_slice_91[400] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_93 = tensor.insert_slice %cst_3 into %inserted_slice_92[416] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_94 = tensor.insert_slice %cst_3 into %inserted_slice_93[432] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_95 = tensor.insert_slice %cst_3 into %inserted_slice_94[448] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_96 = tensor.insert_slice %cst_3 into %inserted_slice_95[464] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_97 = tensor.insert_slice %cst_3 into %inserted_slice_96[480] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_98 = tensor.insert_slice %cst_3 into %inserted_slice_97[496] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_99 = tensor.insert_slice %cst_3 into %inserted_slice_98[512] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_100 = tensor.insert_slice %cst_3 into %inserted_slice_99[528] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_101 = tensor.insert_slice %cst_3 into %inserted_slice_100[544] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_102 = tensor.insert_slice %cst_3 into %inserted_slice_101[560] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_103 = tensor.insert_slice %cst_3 into %inserted_slice_102[576] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_104 = tensor.insert_slice %cst_3 into %inserted_slice_103[592] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_105 = tensor.insert_slice %cst_3 into %inserted_slice_104[608] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_106 = tensor.insert_slice %cst_3 into %inserted_slice_105[624] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_107 = tensor.insert_slice %cst_3 into %inserted_slice_106[640] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_108 = tensor.insert_slice %cst_3 into %inserted_slice_107[656] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_109 = tensor.insert_slice %cst_3 into %inserted_slice_108[672] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_110 = tensor.insert_slice %cst_3 into %inserted_slice_109[688] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_111 = tensor.insert_slice %cst_3 into %inserted_slice_110[704] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_112 = tensor.insert_slice %cst_3 into %inserted_slice_111[720] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_113 = tensor.insert_slice %cst_3 into %inserted_slice_112[736] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_114 = tensor.insert_slice %cst_3 into %inserted_slice_113[752] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_115 = tensor.insert_slice %cst_3 into %inserted_slice_114[768] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_116 = tensor.insert_slice %cst_3 into %inserted_slice_115[784] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_117 = tensor.insert_slice %cst_3 into %inserted_slice_116[800] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_118 = tensor.insert_slice %cst_3 into %inserted_slice_117[816] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_119 = tensor.insert_slice %cst_3 into %inserted_slice_118[832] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_120 = tensor.insert_slice %cst_3 into %inserted_slice_119[848] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_121 = tensor.insert_slice %cst_3 into %inserted_slice_120[864] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_122 = tensor.insert_slice %cst_3 into %inserted_slice_121[880] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_123 = tensor.insert_slice %cst_3 into %inserted_slice_122[896] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_124 = tensor.insert_slice %cst_3 into %inserted_slice_123[912] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_125 = tensor.insert_slice %cst_3 into %inserted_slice_124[928] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_126 = tensor.insert_slice %cst_3 into %inserted_slice_125[944] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_127 = tensor.insert_slice %cst_3 into %inserted_slice_126[960] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_128 = tensor.insert_slice %cst_3 into %inserted_slice_127[976] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_129 = tensor.insert_slice %cst_3 into %inserted_slice_128[992] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %inserted_slice_130 = tensor.insert_slice %cst_3 into %inserted_slice_129[1008] [16] [1] : tensor<16xf32> into tensor<1024xf32>
    %ct_131 = ckks.mul %ct, %ct : (!ct_L2, !ct_L2) -> !ct_L2_D3
    %1 = affine.for %arg0 = 0 to 1024 iter_args(%arg1 = %cst_1) -> (tensor<1024xf32>) {
      %extracted = tensor.extract %inserted_slice_130[%arg0] : tensor<1024xf32>
      %3 = arith.remsi %arg0, %c1024 : index
      %4 = arith.cmpi slt, %3, %c0 : index
      %5 = arith.addi %3, %c1024 : index
      %6 = arith.select %4, %5, %3 : index
      %inserted = tensor.insert %extracted into %arg1[%6] : tensor<1024xf32>
      affine.yield %inserted : tensor<1024xf32>
    }
    %ct_132 = ckks.mul %ct, %ct_0 : (!ct_L2, !ct_L2) -> !ct_L2_D3
    %ct_133 = ckks.relinearize %ct_132 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct_L2_D3 -> !ct_L2_1
    %ct_134 = ckks.rescale %ct_133 {to_ring = #ring_rns_L1_1_x1024} : !ct_L2_1 -> !ct_L1
    %pt = lwe.rlwe_encode %1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %ct_135 = ckks.mul_plain %ct_134, %pt : (!ct_L1, !pt) -> !ct_L1_1
    %pt_136 = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %ct_137 = ckks.mul_plain %ct_131, %pt_136 : (!ct_L2_D3, !pt) -> !ct_L2_D3_1
    %ct_138 = ckks.relinearize %ct_137 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct_L2_D3_1 -> !ct_L2_2
    %ct_139 = ckks.rescale %ct_138 {to_ring = #ring_rns_L1_1_x1024} : !ct_L2_2 -> !ct_L1_1
    %ct_140 = ckks.add %ct_139, %ct_135 : (!ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %pt_141 = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt1
    %ct_142 = ckks.mul_plain %ct_0, %pt_141 : (!ct_L2, !pt1) -> !ct_L2_2
    %ct_143 = ckks.rescale %ct_142 {to_ring = #ring_rns_L1_1_x1024} : !ct_L2_2 -> !ct_L1_1
    %ct_144 = ckks.add %ct_140, %ct_143 : (!ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %2 = affine.for %arg0 = 0 to 1024 iter_args(%arg1 = %cst_1) -> (tensor<1024xf32>) {
      %extracted = tensor.extract %inserted_slice_66[%arg0] : tensor<1024xf32>
      %3 = arith.remsi %arg0, %c1024 : index
      %4 = arith.cmpi slt, %3, %c0 : index
      %5 = arith.addi %3, %c1024 : index
      %6 = arith.select %4, %5, %3 : index
      %inserted = tensor.insert %extracted into %arg1[%6] : tensor<1024xf32>
      affine.yield %inserted : tensor<1024xf32>
    }
    %pt_145 = lwe.rlwe_encode %2 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt1
    %ct_146 = ckks.sub_plain %ct_144, %pt_145 : (!ct_L1_1, !pt1) -> !ct_L1_1
    %ct_147 = ckks.rescale %ct_146 {to_ring = #ring_rns_L0_1_x1024} : !ct_L1_1 -> !ct_L0
    return %ct_147 : !ct_L0
  }
  func.func @polynomial_evaluation__encrypt__arg0(%arg0: tensor<16xf32>, %pk: !pkey_L2) -> !ct_L2 attributes {client.enc_func = {func_name = "polynomial_evaluation", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %concat = tensor.concat dim(0) %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1024xf32>
    %0 = affine.for %arg1 = 0 to 1024 iter_args(%arg2 = %cst) -> (tensor<1024xf32>) {
      %extracted = tensor.extract %concat[%arg1] : tensor<1024xf32>
      %1 = arith.remsi %arg1, %c1024 : index
      %2 = arith.cmpi slt, %1, %c0 : index
      %3 = arith.addi %1, %c1024 : index
      %4 = arith.select %2, %3, %1 : index
      %inserted = tensor.insert %extracted into %arg2[%4] : tensor<1024xf32>
      affine.yield %inserted : tensor<1024xf32>
    }
    %pt = lwe.rlwe_encode %0 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L2) -> !ct_L2
    return %ct : !ct_L2
  }
  func.func @polynomial_evaluation__encrypt__arg1(%arg0: tensor<16xf32>, %pk: !pkey_L2) -> !ct_L2 attributes {client.enc_func = {func_name = "polynomial_evaluation", index = 1 : i64}} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %concat = tensor.concat dim(0) %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1024xf32>
    %0 = affine.for %arg1 = 0 to 1024 iter_args(%arg2 = %cst) -> (tensor<1024xf32>) {
      %extracted = tensor.extract %concat[%arg1] : tensor<1024xf32>
      %1 = arith.remsi %arg1, %c1024 : index
      %2 = arith.cmpi slt, %1, %c0 : index
      %3 = arith.addi %1, %c1024 : index
      %4 = arith.select %2, %3, %1 : index
      %inserted = tensor.insert %extracted into %arg2[%4] : tensor<1024xf32>
      affine.yield %inserted : tensor<1024xf32>
    }
    %pt = lwe.rlwe_encode %0 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L2) -> !ct_L2
    return %ct : !ct_L2
  }
  func.func @polynomial_evaluation__decrypt__result0(%ct: !ct_L0, %sk: !skey_L0) -> tensor<16xf32> attributes {client.dec_func = {func_name = "polynomial_evaluation", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %pt = lwe.rlwe_decrypt %ct, %sk : (!ct_L0, !skey_L0) -> !pt
    %0 = lwe.rlwe_decode %pt {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : !pt -> tensor<1024xf32>
    %1 = affine.for %arg0 = 0 to 1024 iter_args(%arg1 = %cst) -> (tensor<1024xf32>) {
      %2 = arith.remsi %arg0, %c1024 : index
      %3 = arith.cmpi slt, %2, %c0 : index
      %4 = arith.addi %2, %c1024 : index
      %5 = arith.select %3, %4, %2 : index
      %extracted = tensor.extract %0[%5] : tensor<1024xf32>
      %inserted = tensor.insert %extracted into %arg1[%arg0] : tensor<1024xf32>
      affine.yield %inserted : tensor<1024xf32>
    }
    %extracted_slice = tensor.extract_slice %1[0] [16] [1] : tensor<1024xf32> to tensor<16xf32>
    return %extracted_slice : tensor<16xf32>
  }
}
