// RUN: heir-opt --split-preprocessing %s | FileCheck %s

// Tests that an op used for both pre-processing and pre-processed operations
// is duplicated.

// CHECK: func.func @constant__preprocessing
// CHECK-NEXT: arith.constant 0 : index

// CHECK: func.func @constant__preprocessed
// CHECK-NEXT: arith.constant 0 : index

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

func.func @constant(%ct: tensor<2x!ct_L1>, %cleartext: tensor<1x1024xf32>) -> (tensor<1x!ct_L1>) {
  %c0 = arith.constant 0 : index
  %slice = tensor.extract_slice %cleartext [%c0, %c0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
  %pt = lwe.rlwe_encode %slice {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
  %pt_tensor = tensor.from_elements %pt : tensor<1x!pt>
  %ct_slice = tensor.extract_slice %ct [%c0] [1] [1] : tensor<2x!ct_L1> to tensor<1x!ct_L1>
  %0 = ckks.add_plain %ct_slice, %pt_tensor : (tensor<1x!ct_L1>, tensor<1x!pt>) -> tensor<1x!ct_L1>
  return %0 : tensor<1x!ct_L1>
}
