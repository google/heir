// RUN: heir-opt --lwe-to-openfhe %s | FileCheck %s

// CHECK-NOT: lwe

!Z35184269590529_i64 = !mod_arith.int<35184269590529 : i64>
!Z35184270114817_i64 = !mod_arith.int<35184270114817 : i64>
!Z35184272736257_i64 = !mod_arith.int<35184272736257 : i64>
!Z35184275619841_i64 = !mod_arith.int<35184275619841 : i64>
!Z35184279552001_i64 = !mod_arith.int<35184279552001 : i64>
!Z35184281387009_i64 = !mod_arith.int<35184281387009 : i64>
!Z35184284270593_i64 = !mod_arith.int<35184284270593 : i64>
!Z35184290562049_i64 = !mod_arith.int<35184290562049 : i64>
!Z35184297639937_i64 = !mod_arith.int<35184297639937 : i64>
!Z35184301047809_i64 = !mod_arith.int<35184301047809 : i64>
!Z35184306290689_i64 = !mod_arith.int<35184306290689 : i64>
!Z35184307077121_i64 = !mod_arith.int<35184307077121 : i64>
!Z35184314941441_i64 = !mod_arith.int<35184314941441 : i64>
!Z35184316776449_i64 = !mod_arith.int<35184316776449 : i64>
!Z35184318087169_i64 = !mod_arith.int<35184318087169 : i64>
!Z35184320708609_i64 = !mod_arith.int<35184320708609 : i64>
!Z35184329097217_i64 = !mod_arith.int<35184329097217 : i64>
!Z35184330145793_i64 = !mod_arith.int<35184330145793 : i64>
!Z35184339320833_i64 = !mod_arith.int<35184339320833 : i64>
!Z35184345088001_i64 = !mod_arith.int<35184345088001 : i64>
!Z35184350330881_i64 = !mod_arith.int<35184350330881 : i64>
!Z35184365273089_i64 = !mod_arith.int<35184365273089 : i64>
!Z35184376545281_i64 = !mod_arith.int<35184376545281 : i64>
!Z35184377331713_i64 = !mod_arith.int<35184377331713 : i64>
!Z35184385196033_i64 = !mod_arith.int<35184385196033 : i64>
!Z35184399351809_i64 = !mod_arith.int<35184399351809 : i64>
!Z35184404070401_i64 = !mod_arith.int<35184404070401 : i64>
!Z35184410361857_i64 = !mod_arith.int<35184410361857 : i64>
!Z35184414031873_i64 = !mod_arith.int<35184414031873 : i64>
!Z35184415080449_i64 = !mod_arith.int<35184415080449 : i64>
!Z35184415866881_i64 = !mod_arith.int<35184415866881 : i64>
!Z35184423731201_i64 = !mod_arith.int<35184423731201 : i64>
!Z35184430022657_i64 = !mod_arith.int<35184430022657 : i64>
!Z35184430809089_i64 = !mod_arith.int<35184430809089 : i64>
!Z35184436314113_i64 = !mod_arith.int<35184436314113 : i64>
!Z35184440246273_i64 = !mod_arith.int<35184440246273 : i64>
!Z35184440770561_i64 = !mod_arith.int<35184440770561 : i64>
!Z35184446537729_i64 = !mod_arith.int<35184446537729 : i64>
!Z35184452567041_i64 = !mod_arith.int<35184452567041 : i64>
!Z35184454402049_i64 = !mod_arith.int<35184454402049 : i64>
!Z35184454926337_i64 = !mod_arith.int<35184454926337 : i64>
!Z35184463839233_i64 = !mod_arith.int<35184463839233 : i64>
!Z35184465412097_i64 = !mod_arith.int<35184465412097 : i64>
!Z35184474587137_i64 = !mod_arith.int<35184474587137 : i64>
!Z35184478519297_i64 = !mod_arith.int<35184478519297 : i64>
!Z36028797005856769_i64 = !mod_arith.int<36028797005856769 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 9 and 0 <= slot <= 1023 }">
#modulus_chain_L45_C45 = #lwe.modulus_chain<elements = <36028797005856769 : i64, 35184478519297 : i64, 35184269590529 : i64, 35184474587137 : i64, 35184270114817 : i64, 35184465412097 : i64, 35184272736257 : i64, 35184463839233 : i64, 35184275619841 : i64, 35184454926337 : i64, 35184279552001 : i64, 35184454402049 : i64, 35184281387009 : i64, 35184452567041 : i64, 35184284270593 : i64, 35184446537729 : i64, 35184290562049 : i64, 35184440770561 : i64, 35184297639937 : i64, 35184440246273 : i64, 35184301047809 : i64, 35184436314113 : i64, 35184306290689 : i64, 35184430809089 : i64, 35184307077121 : i64, 35184430022657 : i64, 35184314941441 : i64, 35184423731201 : i64, 35184316776449 : i64, 35184415866881 : i64, 35184318087169 : i64, 35184415080449 : i64, 35184320708609 : i64, 35184414031873 : i64, 35184329097217 : i64, 35184410361857 : i64, 35184330145793 : i64, 35184404070401 : i64, 35184339320833 : i64, 35184399351809 : i64, 35184345088001 : i64, 35184385196033 : i64, 35184350330881 : i64, 35184377331713 : i64, 35184365273089 : i64, 35184376545281 : i64>, current = 45>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L45 = !rns.rns<!Z36028797005856769_i64, !Z35184478519297_i64, !Z35184269590529_i64, !Z35184474587137_i64, !Z35184270114817_i64, !Z35184465412097_i64, !Z35184272736257_i64, !Z35184463839233_i64, !Z35184275619841_i64, !Z35184454926337_i64, !Z35184279552001_i64, !Z35184454402049_i64, !Z35184281387009_i64, !Z35184452567041_i64, !Z35184284270593_i64, !Z35184446537729_i64, !Z35184290562049_i64, !Z35184440770561_i64, !Z35184297639937_i64, !Z35184440246273_i64, !Z35184301047809_i64, !Z35184436314113_i64, !Z35184306290689_i64, !Z35184430809089_i64, !Z35184307077121_i64, !Z35184430022657_i64, !Z35184314941441_i64, !Z35184423731201_i64, !Z35184316776449_i64, !Z35184415866881_i64, !Z35184318087169_i64, !Z35184415080449_i64, !Z35184320708609_i64, !Z35184414031873_i64, !Z35184329097217_i64, !Z35184410361857_i64, !Z35184330145793_i64, !Z35184404070401_i64, !Z35184339320833_i64, !Z35184399351809_i64, !Z35184345088001_i64, !Z35184385196033_i64, !Z35184350330881_i64, !Z35184377331713_i64, !Z35184365273089_i64, !Z35184376545281_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<1x10xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L45_1_x1024 = #polynomial.ring<coefficientType = !rns_L45, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L45 = #lwe.ciphertext_space<ring = #ring_rns_L45_1_x1024, encryption_type = mix>
!ct_L45 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L45, key = #key, modulus_chain = #modulus_chain_L45_C45>
module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 17, Q = [36028797005856769, 35184478519297, 35184269590529, 35184474587137, 35184270114817, 35184465412097, 35184272736257, 35184463839233, 35184275619841, 35184454926337, 35184279552001, 35184454402049, 35184281387009, 35184452567041, 35184284270593, 35184446537729, 35184290562049, 35184440770561, 35184297639937, 35184440246273, 35184301047809, 35184436314113, 35184306290689, 35184430809089, 35184307077121, 35184430022657, 35184314941441, 35184423731201, 35184316776449, 35184415866881, 35184318087169, 35184415080449, 35184320708609, 35184414031873, 35184329097217, 35184410361857, 35184330145793, 35184404070401, 35184339320833, 35184399351809, 35184345088001, 35184385196033, 35184350330881, 35184377331713, 35184365273089, 35184376545281], P = [1152921504616808449, 1152921504618381313, 1152921504622575617, 1152921504634109953, 1152921504643809281, 1152921504650100737, 1152921504663994369, 1152921504666615809, 1152921504672645121, 1152921504687849473, 1152921504690208769, 1152921504690995201], logDefaultScale = 45>, scheme.ckks} {
  func.func @lenet(%arg0: tensor<1x!ct_L45> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x32x32xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-32i2 - i3 + slot) mod 1024 = 0 and 0 <= i2 <= 31 and 0 <= i3 <= 31 and 0 <= slot <= 1023 }">>}, %arg1: tensor<1x!ct_L45>, %arg2: tensor<1x!ct_L45>, %arg3: tensor<1x!ct_L45>, %arg4: tensor<1x!ct_L45>, %arg5: tensor<1x!ct_L45>, %arg6: tensor<1x1024xf32>, %ct: !ct_L45, %pt: !pt, %arg7: tensor<1x!ct_L45>) -> (tensor<1x!ct_L45> {tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense<5.000000e+00> : tensor<6x1024xf32>
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 0.235571414 : f32
    %0 = tensor.empty() : tensor<6x!ct_L45>
    %c1_i32 = arith.constant 1 : i32
    %c784_i32 = arith.constant 784 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    %inserted_slice = tensor.insert_slice %arg1 into %0[0] [1] [1] : tensor<1x!ct_L45> into tensor<6x!ct_L45>
    %inserted_slice_2 = tensor.insert_slice %arg2 into %inserted_slice[1] [1] [1] : tensor<1x!ct_L45> into tensor<6x!ct_L45>
    %inserted_slice_3 = tensor.insert_slice %arg3 into %inserted_slice_2[2] [1] [1] : tensor<1x!ct_L45> into tensor<6x!ct_L45>
    %inserted_slice_4 = tensor.insert_slice %arg4 into %inserted_slice_3[3] [1] [1] : tensor<1x!ct_L45> into tensor<6x!ct_L45>
    %inserted_slice_5 = tensor.insert_slice %arg5 into %inserted_slice_4[4] [1] [1] : tensor<1x!ct_L45> into tensor<6x!ct_L45>
    %ct_6 = lwe.rmul_plain %ct, %pt : (!ct_L45, !pt) -> !ct_L45
    %1 = arith.mulf %arg6, %cst fastmath<nnan,nsz> : tensor<1x1024xf32>
    %extracted_slice = tensor.extract_slice %1[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_7 = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %ct_8 = lwe.radd_plain %ct_6, %pt_7 : (!ct_L45, !pt) -> !ct_L45
    %inserted = tensor.insert %ct_8 into %arg7[%c0] : tensor<1x!ct_L45>
    %inserted_slice_9 = tensor.insert_slice %inserted into %inserted_slice_5[5] [1] [1] : tensor<1x!ct_L45> into tensor<6x!ct_L45>
    %2 = scf.for %arg8 = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%arg9 = %cst_0) -> (tensor<6x1024xf32>)  : i32 {
      %4 = scf.for %arg10 = %c0_i32 to %c784_i32 step %c1_i32 iter_args(%arg11 = %arg9) -> (tensor<6x1024xf32>)  : i32 {
        %5 = arith.index_cast %arg8 : i32 to index
        %6 = arith.index_cast %arg10 : i32 to index
        %inserted_23 = tensor.insert %cst_1 into %arg11[%5, %6] : tensor<6x1024xf32>
        scf.yield %inserted_23 : tensor<6x1024xf32>
      }
      scf.yield %4 : tensor<6x1024xf32>
    }
    %extracted_slice_10 = tensor.extract_slice %2[0, 0] [1, 1024] [1, 1] : tensor<6x1024xf32> to tensor<1024xf32>
    %pt_11 = lwe.rlwe_encode %extracted_slice_10 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %extracted_slice_12 = tensor.extract_slice %2[1, 0] [1, 1024] [1, 1] : tensor<6x1024xf32> to tensor<1024xf32>
    %pt_13 = lwe.rlwe_encode %extracted_slice_12 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %extracted_slice_14 = tensor.extract_slice %2[2, 0] [1, 1024] [1, 1] : tensor<6x1024xf32> to tensor<1024xf32>
    %pt_15 = lwe.rlwe_encode %extracted_slice_14 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %extracted_slice_16 = tensor.extract_slice %2[3, 0] [1, 1024] [1, 1] : tensor<6x1024xf32> to tensor<1024xf32>
    %pt_17 = lwe.rlwe_encode %extracted_slice_16 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %extracted_slice_18 = tensor.extract_slice %2[4, 0] [1, 1024] [1, 1] : tensor<6x1024xf32> to tensor<1024xf32>
    %pt_19 = lwe.rlwe_encode %extracted_slice_18 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %extracted_slice_20 = tensor.extract_slice %2[5, 0] [1, 1024] [1, 1] : tensor<6x1024xf32> to tensor<1024xf32>
    %pt_21 = lwe.rlwe_encode %extracted_slice_20 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements = tensor.from_elements %pt_11, %pt_13, %pt_15, %pt_17, %pt_19, %pt_21 : tensor<6x!pt>
    %3 = affine.for %arg8 = 0 to 6 iter_args(%arg9 = %0) -> (tensor<6x!ct_L45>) {
      %extracted_23 = tensor.extract %inserted_slice_9[%arg8] : tensor<6x!ct_L45>
      %extracted_24 = tensor.extract %from_elements[%arg8] : tensor<6x!pt>
      %ct_25 = lwe.rmul_plain %extracted_23, %extracted_24 : (!ct_L45, !pt) -> !ct_L45
      %inserted_26 = tensor.insert %ct_25 into %arg9[%arg8] : tensor<6x!ct_L45>
      affine.yield %inserted_26 : tensor<6x!ct_L45>
    }
    %extracted = tensor.extract %3[%c0] : tensor<6x!ct_L45>
    %from_elements_22 = tensor.from_elements %extracted : tensor<1x!ct_L45>
    return %from_elements_22 : tensor<1x!ct_L45>
  }
}
