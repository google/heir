// RUN: heir-opt %s --split-preprocessing

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 90>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 32 = 0 and 0 <= i0 <= 31 and 0 <= slot <= 1023 }">
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!Z36028797018652673_i64>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<32xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pkey_L0 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L0_1_x1024>
!pkey_L1 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>
!skey_L0 = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x1024>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.actual_slot_count = 4096 : i64, scheme.ckks, scheme.requested_slot_count = 1024 : i64} {
  func.func private @_assign_layout_11730091706342691187() -> tensor<32x1024xf32> attributes {client.pack_func = {func_name = "matvec"}} {
    %cst = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x1024xf32>
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = scf.for %arg0 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg1 = %cst_0) -> (tensor<32x1024xf32>)  : i32 {
      %1 = scf.for %arg2 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<32x1024xf32>)  : i32 {
        %2 = arith.addi %arg0, %arg2 : i32
        %3 = arith.addi %2, %c31_i32 : i32
        %4 = arith.remsi %3, %c64_i32 : i32
        %5 = arith.cmpi sge, %4, %c31_i32 : i32
        %6 = scf.if %5 -> (tensor<32x1024xf32>) {
          %7 = arith.index_cast %arg0 : i32 to index
          %8 = arith.index_cast %arg2 : i32 to index
          %inserted = tensor.insert %cst into %arg3[%7, %8] : tensor<32x1024xf32>
          scf.yield %inserted : tensor<32x1024xf32>
        } else {
          scf.yield %arg3 : tensor<32x1024xf32>
        }
        scf.yield %6 : tensor<32x1024xf32>
      }
      scf.yield %1 : tensor<32x1024xf32>
    }
    return %0 : tensor<32x1024xf32>
  }
  func.func @matvec(%arg0: tensor<1x!ct_L1> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<33xf32>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 64 = 0 and 0 <= i0 <= 32 and 0 <= slot <= 1023 }">>}, %ct: !ct_L0 {client.enc_zero_arg}) -> (tensor<1x!ct_L0> {tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c-6 = arith.constant -6 : index
    %0 = call @_assign_layout_11730091706342691187() : () -> tensor<32x1024xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<32x1024xf32> to tensor<1x1024xf32>
    %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %pt = lwe.rlwe_encode %extracted_slice_1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %1 = ckks.mul_plain %from_elements, %arg0 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1_1>
    %2 = ckks.rescale %1 {to_ring = #ring_rns_L0_1_x1024} : tensor<1x!ct_L1_1> -> tensor<1x!ct_L0>
    %pt_2 = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %from_elements_3 = tensor.from_elements %pt_2 : tensor<1x!pt>
    %3 = ckks.add_plain %from_elements_3, %2 : (tensor<1x!pt>, tensor<1x!ct_L0>) -> tensor<1x!ct_L0>
    %4 = scf.for %arg1 = %c1 to %c6 step %c1 iter_args(%arg2 = %3) -> (tensor<1x!ct_L0>) {
      %9 = arith.cmpi slt, %arg1, %c32 : index
      %10 = scf.if %9 -> (tensor<1x!ct_L0>) {
        %extracted_slice_4 = tensor.extract_slice %0[%arg1, 0] [1, 1024] [1, 1] : tensor<32x1024xf32> to tensor<1x1024xf32>
        %11 = ckks.rotate %arg0, %arg1 : index : tensor<1x!ct_L1>
        %extracted_slice_5 = tensor.extract_slice %extracted_slice_4[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
        %pt_6 = lwe.rlwe_encode %extracted_slice_5 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
        %from_elements_7 = tensor.from_elements %pt_6 : tensor<1x!pt>
        %12 = ckks.mul_plain %from_elements_7, %11 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1_1>
        %13 = ckks.rescale %12 {to_ring = #ring_rns_L0_1_x1024} : tensor<1x!ct_L1_1> -> tensor<1x!ct_L0>
        %14 = ckks.add %arg2, %13 : (tensor<1x!ct_L0>, tensor<1x!ct_L0>) -> tensor<1x!ct_L0>
        scf.yield %14 : tensor<1x!ct_L0>
      } else {
        scf.yield %arg2 : tensor<1x!ct_L0>
      }
      scf.yield %10 : tensor<1x!ct_L0>
    }
    %5 = ckks.add_plain %from_elements_3, %4 : (tensor<1x!pt>, tensor<1x!ct_L0>) -> tensor<1x!ct_L0>
    %6 = scf.for %arg1 = %c1 to %c6 step %c1 iter_args(%arg2 = %5) -> (tensor<1x!ct_L0>) {
      %9 = arith.muli %arg1, %c6 : index
      %10 = arith.cmpi slt, %9, %c32 : index
      %11 = scf.if %10 -> (tensor<1x!ct_L0>) {
        %extracted_slice_4 = tensor.extract_slice %0[%9, 0] [1, 1024] [1, 1] : tensor<32x1024xf32> to tensor<1x1024xf32>
        %15 = arith.muli %arg1, %c-6 : index
        %16 = tensor_ext.rotate %extracted_slice_4, %15 : tensor<1x1024xf32>, index
        %extracted_slice_5 = tensor.extract_slice %16[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
        %pt_6 = lwe.rlwe_encode %extracted_slice_5 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
        %from_elements_7 = tensor.from_elements %pt_6 : tensor<1x!pt>
        %17 = ckks.mul_plain %from_elements_7, %arg0 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1_1>
        %18 = ckks.rescale %17 {to_ring = #ring_rns_L0_1_x1024} : tensor<1x!ct_L1_1> -> tensor<1x!ct_L0>
        %19 = ckks.add_plain %from_elements_3, %18 : (tensor<1x!pt>, tensor<1x!ct_L0>) -> tensor<1x!ct_L0>
        scf.yield %19 : tensor<1x!ct_L0>
      } else {
        %15 = lwe.rlwe_encode %cst_0 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1x1024xf32> -> tensor<1x!pt>
        %splat = tensor.splat %ct : tensor<1x!ct_L0>
        %16 = lwe.radd_plain %splat, %15 : (tensor<1x!ct_L0>, tensor<1x!pt>) -> tensor<1x!ct_L0>
        scf.yield %16 : tensor<1x!ct_L0>
      }
      %12 = scf.for %arg3 = %c1 to %c6 step %c1 iter_args(%arg4 = %11) -> (tensor<1x!ct_L0>) {
        %15 = arith.addi %arg3, %9 : index
        %16 = arith.cmpi slt, %15, %c32 : index
        %17 = scf.if %16 -> (tensor<1x!ct_L0>) {
          %extracted_slice_4 = tensor.extract_slice %0[%15, 0] [1, 1024] [1, 1] : tensor<32x1024xf32> to tensor<1x1024xf32>
          %18 = arith.muli %arg1, %c-6 : index
          %19 = tensor_ext.rotate %extracted_slice_4, %18 : tensor<1x1024xf32>, index
          %20 = ckks.rotate %arg0, %arg3 : index : tensor<1x!ct_L1>
          %extracted_slice_5 = tensor.extract_slice %19[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
          %pt_6 = lwe.rlwe_encode %extracted_slice_5 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
          %from_elements_7 = tensor.from_elements %pt_6 : tensor<1x!pt>
          %21 = ckks.mul_plain %from_elements_7, %20 : (tensor<1x!pt>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1_1>
          %22 = ckks.rescale %21 {to_ring = #ring_rns_L0_1_x1024} : tensor<1x!ct_L1_1> -> tensor<1x!ct_L0>
          %23 = ckks.add %arg4, %22 : (tensor<1x!ct_L0>, tensor<1x!ct_L0>) -> tensor<1x!ct_L0>
          scf.yield %23 : tensor<1x!ct_L0>
        } else {
          scf.yield %arg4 : tensor<1x!ct_L0>
        }
        scf.yield %17 : tensor<1x!ct_L0>
      }
      %13 = ckks.rotate %12, %9 : index : tensor<1x!ct_L0>
      %14 = ckks.add %arg2, %13 : (tensor<1x!ct_L0>, tensor<1x!ct_L0>) -> tensor<1x!ct_L0>
      scf.yield %14 : tensor<1x!ct_L0>
    }
    %7 = ckks.rotate %6, %c32 : index : tensor<1x!ct_L0>
    %8 = ckks.add %6, %7 : (tensor<1x!ct_L0>, tensor<1x!ct_L0>) -> tensor<1x!ct_L0>
    return %8 : tensor<1x!ct_L0>
  }
}
