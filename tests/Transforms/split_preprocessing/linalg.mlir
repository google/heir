// RUN: heir-opt --split-preprocessing %s | FileCheck %s

// Tests that a linalg op can be moved into preprocessing. This ensures that ops
// inside linalg op regions aren't double-cloned.

// CHECK-DAG: ![[pt:.*]] = !lwe.lwe_plaintext
// CHECK-DAG: ![[ct_L1:.*]] = !lwe.lwe_ciphertext

// CHECK: func.func @linalg__preprocessing() -> tensor<1x![[pt]]>
// CHECK-SAME: client.pack_func = {func_name = "linalg"}
// CHECK: linalg.broadcast
// CHECK: lwe.rlwe_encode

// CHECK: func.func @linalg__preprocessed(%[[ct:.*]]: ![[ct_L1]], %[[arg0:.*]]: tensor<1x![[pt]]>) -> ![[ct_L1]]
// CHECK-SAME: client.preprocessed_func = {func_name = "linalg"}

// CHECK: func.func @linalg
// CHECK-SAME: (%[[CT:.*]]: ![[ct_L1]])
// CHECK-NEXT:   %[[PT:.*]] = call @linalg__preprocessing()
// CHECK-NEXT:   %[[CALL:.*]] = call @linalg__preprocessed(%[[CT]], %[[PT]]) : (![[ct_L1]], tensor<1x![[pt]]>) -> ![[ct_L1]]
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
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pkey_L1 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>
!skey_L1 = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x1024>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

func.func @linalg(%ct: !ct_L1) -> (!ct_L1) {
  %c1 = arith.constant dense<1.0> : tensor<f32>
  %0 = tensor.empty() : tensor<1024xf32>
  %c2 = linalg.broadcast ins(%c1 : tensor<f32>) outs(%0 : tensor<1024xf32>) dimensions = [0]
  %pt1 = lwe.rlwe_encode %c2 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
  %1 = ckks.add_plain %ct, %pt1 : (!ct_L1, !pt) -> !ct_L1
  return %1 : !ct_L1
}
