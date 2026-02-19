// RUN: heir-opt --lwe-to-openfhe %s | FileCheck %s

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 512 = 0 and 0 <= i1 <= 511 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 512 = 0 and (-i1 + ct + slot) mod 1024 = 0 and 0 <= i0 <= 511 and 0 <= i1 <= 783 and 0 <= ct <= 511 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<1x512xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

// CHECK-DAG: [[ccType:.*]] = !openfhe.crypto_context
// CHECK-DAG: [[ctType:.*]] = !openfhe.ciphertext
// CHECK-DAG: [[ptType:.*]] = !openfhe.plaintext
module @jit_func attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>, jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, scheme.ckks} {
  // CHECK: func.func private @_assign_layout_11979326689855340354(tensor<512x784xf32>) -> tensor<512x1024xf32>
  // CHECK: func.func @mnist__preprocessed(%[[cc:.*]]: [[ccType]],
  // CHECK: func.func public @mnist(%[[cc:.*]]: [[ccType]],
  // CHECK:   call @mnist__preprocessed(%[[cc]],
  func.func private @_assign_layout_11979326689855340354(tensor<512x784xf32>) -> tensor<512x1024xf32> attributes {client.pack_func = {func_name = "mnist"}}
  func.func @mnist__preprocessed(%arg0: tensor<512x1024xf32> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<512x784xf32>, layout = #layout1>}, %arg1: tensor<1x!ct_L1> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x784xf32>, layout = #layout2>}) -> (tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) attributes {client.preprocessed_func = {func_name = "mnist"}} {
    %c0 = arith.constant 0 : index
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1024xf32>
    %pt = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    %extracted = tensor.extract %arg1[%c0] : tensor<1x!ct_L1>
    %ct = lwe.rmul_plain %extracted, %pt : (!ct_L1, !pt) -> !ct_L1
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L1>
    return %from_elements : tensor<1x!ct_L1>
  }
  func.func public @mnist(%arg0: tensor<512x784xf32>, %arg1: tensor<1x!ct_L1> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x784xf32>, layout = #layout2>}) -> (tensor<1x!ct_L1> {jax.result_info = "result[0]", tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = call @_assign_layout_11979326689855340354(%arg0) : (tensor<512x784xf32>) -> tensor<512x1024xf32>
    %1 = call @mnist__preprocessed(%0, %arg1) {arg_attrs = [{mhlo.sharding = "{replicated}", tensor_ext.layout = #layout1}, {tensor_ext.layout = #layout2}]} : (tensor<512x1024xf32>, tensor<1x!ct_L1>) -> tensor<1x!ct_L1>
    return %1 : tensor<1x!ct_L1>
  }
}
