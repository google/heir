// RUN: heir-opt --split-preprocessing %s

// CHECK: func.func @matvec
// CHECK: call @matvec__preprocessing
// CHECK-SAME: -> (tensor<2x!pt>, tensor<2x!pt>, tensor<2x!pt>, tensor<2x!pt>, tensor<2x!pt>, tensor<2x!pt>, tensor<2x!pt>, tensor<2x!pt>, tensor<1x!pt>)
// CHECK: call @matvec__preprocessed

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 1023 }">
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<16xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pkey_L1 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>
!skey_L1 = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x1024>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>, scheme.ckks} {
  func.func private @_assign_layout_6046580691004308546(%arg0: tensor<16xf32>) -> tensor<1x1024xf32> attributes {client.pack_func = {func_name = "matvec"}} {
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>)  : i32 {
      %1 = arith.remsi %arg1, %c16_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %extracted = tensor.extract %arg0[%2] : tensor<16xf32>
      %3 = arith.index_cast %arg1 : i32 to index
      %inserted = tensor.insert %extracted into %arg2[%c0, %3] : tensor<1x1024xf32>
      scf.yield %inserted : tensor<1x1024xf32>
    }
    return %0 : tensor<1x1024xf32>
  }
  func.func private @_assign_layout_82497239515497017(%arg0: tensor<16x16xf32>) -> tensor<16x1024xf32> attributes {client.pack_func = {func_name = "matvec"}} {
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<16x1024xf32>)  : i32 {
      %1 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg4 = %arg2) -> (tensor<16x1024xf32>)  : i32 {
        %2 = arith.remsi %arg3, %c16_i32 : i32
        %3 = arith.addi %arg1, %arg3 : i32
        %4 = arith.remsi %3, %c16_i32 : i32
        %5 = arith.index_cast %2 : i32 to index
        %6 = arith.index_cast %4 : i32 to index
        %extracted = tensor.extract %arg0[%5, %6] : tensor<16x16xf32>
        %7 = arith.index_cast %arg1 : i32 to index
        %8 = arith.index_cast %arg3 : i32 to index
        %inserted = tensor.insert %extracted into %arg4[%7, %8] : tensor<16x1024xf32>
        scf.yield %inserted : tensor<16x1024xf32>
      }
      scf.yield %1 : tensor<16x1024xf32>
    }
    return %0 : tensor<16x1024xf32>
  }
  func.func @matvec(%arg0: tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) {
    %c-12 = arith.constant -12 : index
    %c-8 = arith.constant -8 : index
    %c-4 = arith.constant -4 : index
    %cst = arith.constant dense_resource<__elided__> : tensor<16x16xf32>
    %cst_0 = arith.constant dense<[-0.45141533, -0.0277900472, 0.311195374, 0.18254894, -0.258809537, 0.497506738, 0.00115649134, -0.194445714, 0.158549473, 0.000000e+00, 0.310650676, -0.214976981, -0.023661999, -0.392960966, 6.472870e-01, 0.831665277]> : tensor<16xf32>
    %0 = call @_assign_layout_82497239515497017(%cst) : (tensor<16x16xf32>) -> tensor<16x1024xf32>
    %1 = call @_assign_layout_6046580691004308546(%cst_0) : (tensor<16xf32>) -> tensor<1x1024xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_1 = tensor.extract_slice %0[1, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_2 = tensor.extract_slice %0[2, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_3 = tensor.extract_slice %0[3, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_4 = tensor.extract_slice %0[4, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_5 = tensor.extract_slice %0[5, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_6 = tensor.extract_slice %0[6, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_7 = tensor.extract_slice %0[7, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_8 = tensor.extract_slice %0[8, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_9 = tensor.extract_slice %0[9, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_10 = tensor.extract_slice %0[10, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_11 = tensor.extract_slice %0[11, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_12 = tensor.extract_slice %0[12, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_13 = tensor.extract_slice %0[13, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_14 = tensor.extract_slice %0[14, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_15 = tensor.extract_slice %0[15, 0] [1, 1024] [1, 1] : tensor<16x1024xf32> to tensor<1x1024xf32>
    %2 = tensor_ext.rotate %extracted_slice_4, %c-4 : tensor<1x1024xf32>, index
    %3 = tensor_ext.rotate %extracted_slice_5, %c-4 : tensor<1x1024xf32>, index
    %4 = tensor_ext.rotate %extracted_slice_6, %c-4 : tensor<1x1024xf32>, index
    %5 = tensor_ext.rotate %extracted_slice_7, %c-4 : tensor<1x1024xf32>, index
    %6 = tensor_ext.rotate %extracted_slice_8, %c-8 : tensor<1x1024xf32>, index
    %7 = tensor_ext.rotate %extracted_slice_9, %c-8 : tensor<1x1024xf32>, index
    %8 = tensor_ext.rotate %extracted_slice_10, %c-8 : tensor<1x1024xf32>, index
    %9 = tensor_ext.rotate %extracted_slice_11, %c-8 : tensor<1x1024xf32>, index
    %10 = tensor_ext.rotate %extracted_slice_12, %c-12 : tensor<1x1024xf32>, index
    %11 = tensor_ext.rotate %extracted_slice_13, %c-12 : tensor<1x1024xf32>, index
    %12 = tensor_ext.rotate %extracted_slice_14, %c-12 : tensor<1x1024xf32>, index
    %13 = tensor_ext.rotate %extracted_slice_15, %c-12 : tensor<1x1024xf32>, index
    %extracted_slice_16 = tensor.extract_slice %extracted_slice[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt = lwe.rlwe_encode %extracted_slice_16 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %14 = ckks.mul_plain %from_elements, %arg0 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %15 = ckks.rotate %arg0 {static_shift = 1 : index} : tensor<1x!ct_L1>
    %extracted_slice_17 = tensor.extract_slice %extracted_slice_1[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_18 = lwe.rlwe_encode %extracted_slice_17 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_19 = tensor.from_elements %pt_18 : tensor<1x!pt>
    %16 = ckks.mul_plain %from_elements_19, %15 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %17 = ckks.rotate %arg0 {static_shift = 2 : index} : tensor<1x!ct_L1>
    %extracted_slice_20 = tensor.extract_slice %extracted_slice_2[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_21 = lwe.rlwe_encode %extracted_slice_20 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_22 = tensor.from_elements %pt_21 : tensor<1x!pt>
    %18 = ckks.mul_plain %from_elements_22, %17 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %19 = ckks.rotate %arg0 {static_shift = 3 : index} : tensor<1x!ct_L1>
    %extracted_slice_23 = tensor.extract_slice %extracted_slice_3[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_24 = lwe.rlwe_encode %extracted_slice_23 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_25 = tensor.from_elements %pt_24 : tensor<1x!pt>
    %20 = ckks.mul_plain %from_elements_25, %19 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_26 = tensor.extract_slice %2[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_27 = lwe.rlwe_encode %extracted_slice_26 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_28 = tensor.from_elements %pt_27 : tensor<1x!pt>
    %21 = ckks.mul_plain %from_elements_28, %arg0 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_29 = tensor.extract_slice %3[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_30 = lwe.rlwe_encode %extracted_slice_29 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_31 = tensor.from_elements %pt_30 : tensor<1x!pt>
    %22 = ckks.mul_plain %from_elements_31, %15 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_32 = tensor.extract_slice %4[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_33 = lwe.rlwe_encode %extracted_slice_32 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_34 = tensor.from_elements %pt_33 : tensor<1x!pt>
    %23 = ckks.mul_plain %from_elements_34, %17 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_35 = tensor.extract_slice %5[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_36 = lwe.rlwe_encode %extracted_slice_35 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_37 = tensor.from_elements %pt_36 : tensor<1x!pt>
    %24 = ckks.mul_plain %from_elements_37, %19 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %25 = ckks.add %21, %22 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %26 = ckks.add %23, %24 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %27 = ckks.add %25, %26 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %28 = ckks.rotate %27 {static_shift = 4 : index} : tensor<1x!ct_L1>
    %extracted_slice_38 = tensor.extract_slice %6[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_39 = lwe.rlwe_encode %extracted_slice_38 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_40 = tensor.from_elements %pt_39 : tensor<1x!pt>
    %29 = ckks.mul_plain %from_elements_40, %arg0 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_41 = tensor.extract_slice %7[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_42 = lwe.rlwe_encode %extracted_slice_41 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_43 = tensor.from_elements %pt_42 : tensor<1x!pt>
    %30 = ckks.mul_plain %from_elements_43, %15 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_44 = tensor.extract_slice %8[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_45 = lwe.rlwe_encode %extracted_slice_44 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_46 = tensor.from_elements %pt_45 : tensor<1x!pt>
    %31 = ckks.mul_plain %from_elements_46, %17 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_47 = tensor.extract_slice %9[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_48 = lwe.rlwe_encode %extracted_slice_47 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_49 = tensor.from_elements %pt_48 : tensor<1x!pt>
    %32 = ckks.mul_plain %from_elements_49, %19 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %33 = ckks.add %29, %30 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %34 = ckks.add %31, %32 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %35 = ckks.add %33, %34 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %36 = ckks.rotate %35 {static_shift = 8 : index} : tensor<1x!ct_L1>
    %extracted_slice_50 = tensor.extract_slice %10[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_51 = lwe.rlwe_encode %extracted_slice_50 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_52 = tensor.from_elements %pt_51 : tensor<1x!pt>
    %37 = ckks.mul_plain %from_elements_52, %arg0 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_53 = tensor.extract_slice %11[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_54 = lwe.rlwe_encode %extracted_slice_53 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_55 = tensor.from_elements %pt_54 : tensor<1x!pt>
    %38 = ckks.mul_plain %from_elements_55, %15 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_56 = tensor.extract_slice %12[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_57 = lwe.rlwe_encode %extracted_slice_56 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_58 = tensor.from_elements %pt_57 : tensor<1x!pt>
    %39 = ckks.mul_plain %from_elements_58, %17 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_59 = tensor.extract_slice %13[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_60 = lwe.rlwe_encode %extracted_slice_59 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_61 = tensor.from_elements %pt_60 : tensor<1x!pt>
    %40 = ckks.mul_plain %from_elements_61, %19 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %41 = ckks.add %37, %38 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %42 = ckks.add %39, %40 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %43 = ckks.add %41, %42 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %44 = ckks.rotate %43 {static_shift = 12 : index} : tensor<1x!ct_L1>
    %45 = ckks.add %14, %16 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %46 = ckks.add %18, %20 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %47 = ckks.add %45, %46 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %48 = ckks.add %28, %36 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %extracted_slice_62 = tensor.extract_slice %1[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt_63 = lwe.rlwe_encode %extracted_slice_62 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_64 = tensor.from_elements %pt_63 : tensor<1x!pt>
    %49 = ckks.add_plain %44, %from_elements_64 : (tensor<1x!ct_L1>, tensor<1x!pt>) -> tensor<1x!ct_L1>
    %50 = ckks.add %48, %49 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    %51 = ckks.add %47, %50 : (tensor<1x!ct_L1>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    return %51 : tensor<1x!ct_L1>
  }
}
