// RUN: heir-opt --split-preprocessing --split-input-file %s | FileCheck %s

// CHECK-DAG: ![[pt:.*]] = !lwe.lwe_plaintext
// CHECK-DAG: ![[ct_L1:.*]] = !lwe.lwe_ciphertext

// CHECK: func.func @hoist_arg__preprocessing(%[[arg0:.*]]: tensor<1024xf32>) -> ![[pt]]
// CHECK-SAME: client.pack_func = {func_name = "hoist_arg"}

// CHECK: func.func @hoist_arg(
// CHECK-SAME: %[[CT:.*]]: ![[ct_L1]],
// CHECK-SAME: %[[ARG0:.*]]: tensor<1024xf32>)
// CHECK-NEXT:   %[[PT:.*]] = call @hoist_arg__preprocessing(%[[ARG0]])
// CHECK-NEXT:   %[[CALL:.*]] = call @hoist_arg__preprocessed(%[[CT]], %[[PT]])
// CHECK-NEXT:   return %[[CALL]]
// CHECK-NEXT: }

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 1023 }">
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<16xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pkey_L1 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>
!skey_L1 = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x1024>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

func.func @hoist_arg(%ct: !ct_L1, %c1: tensor<1024xf32>) -> (!ct_L1) {
  %pt = lwe.rlwe_encode %c1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
  %0 = ckks.add_plain %ct, %pt : (!ct_L1, !pt) -> !ct_L1
  return %0 : !ct_L1
}

// -----

// CHECK-DAG: ![[pt:.*]] = !lwe.lwe_plaintext
// CHECK-DAG: ![[ct_L1:.*]] = !lwe.lwe_ciphertext

// CHECK: func.func @hoist_arg_and_constant__preprocessing(%[[arg0:.*]]: tensor<1024xf32>) -> (![[pt]], ![[pt]])
// CHECK-SAME: client.pack_func = {func_name = "hoist_arg_and_constant"}

// CHECK: func.func @hoist_arg_and_constant__preprocessed(%[[arg0:.*]]: ![[ct_L1]],
// CHECK-SAME: %[[ARG0:.*]]: ![[pt]],
// CHECK-SAME: %[[ARG1:.*]]: ![[pt]]) -> ![[ct_L1]]
// CHECK-SAME: client.preprocessed_func = {func_name = "hoist_arg_and_constant"}

// CHECK: func.func @hoist_arg_and_constant
// CHECK-SAME: (%[[CT:.*]]: ![[ct_L1]],
// CHECK-SAME: %[[ARG0:.*]]: tensor<1024xf32>)
// CHECK-NEXT:   %[[PT1:.*]], %[[PT2:.*]] = call @hoist_arg_and_constant__preprocessing(%[[ARG0]])
// CHECK-NEXT:   %[[CALL:.*]] = call @hoist_arg_and_constant__preprocessed(%[[CT]], %[[PT1]], %[[PT2]])
// CHECK-NEXT:   return %[[CALL]]
// CHECK-NEXT: }

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 1023 }">
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<16xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pkey_L1 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>
!skey_L1 = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x1024>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

func.func @hoist_arg_and_constant(%ct: !ct_L1, %c1: tensor<1024xf32>) -> (!ct_L1) {
  %c2 = arith.constant dense<2.0> : tensor<1024xf32>
  %pt = lwe.rlwe_encode %c1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
  %pt2 = lwe.rlwe_encode %c2 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
  %0 = ckks.add_plain %ct, %pt : (!ct_L1, !pt) -> !ct_L1
  %1 = ckks.add_plain %ct, %pt2 : (!ct_L1, !pt) -> !ct_L1
  return %1 : !ct_L1
}

// -----

// CHECK-DAG: ![[pt:.*]] = !lwe.lwe_plaintext
// CHECK-DAG: ![[ct_L1:.*]] = !lwe.lwe_ciphertext

// CHECK: func.func @hoist_with_computation__preprocessing(%[[arg0:.*]]: tensor<1x1024xf32>) -> ![[pt]]
// CHECK-SAME: client.pack_func = {func_name = "hoist_with_computation"}
// CHECK-NEXT: tensor.extract_slice

// CHECK: func.func @hoist_with_computation__preprocessed(%[[arg0:.*]]: ![[ct_L1]],
// CHECK-SAME: %[[ARG0:.*]]: ![[pt]])
// CHECK-SAME: client.preprocessed_func = {func_name = "hoist_with_computation"}

// CHECK: func.func @hoist_with_computation
// CHECK-SAME: (%[[CT:.*]]: ![[ct_L1]],
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x1024xf32>)
// CHECK-NEXT:   %[[PT:.*]] = call @hoist_with_computation__preprocessing(%[[ARG0]])
// CHECK-NEXT:   %[[CALL:.*]] = call @hoist_with_computation__preprocessed(%[[CT]], %[[PT]])
// CHECK-NEXT:   return %[[CALL]]
// CHECK-NEXT: }

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 1023 }">
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<16xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pkey_L1 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>
!skey_L1 = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x1024>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<1024xf32>>, plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

func.func @hoist_with_computation(%ct: !ct_L1, %tensorC1: tensor<1x1024xf32>) -> (!ct_L1) {
  %c1 = tensor.extract_slice %tensorC1[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
  %pt = lwe.rlwe_encode %c1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
  %0 = ckks.add_plain %ct, %pt : (!ct_L1, !pt) -> !ct_L1
  return %0 : !ct_L1
}
