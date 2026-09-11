// RUN: heir-opt --split-preprocessing %s | FileCheck %s

// CHECK-DAG: ![[pt:.*]] = !lwe.lwe_plaintext
// CHECK-DAG: ![[ct_L1:.*]] = !lwe.lwe_ciphertext

// CHECK: func.func @hoist_one_assign__preprocessing() -> !preprocessing.storage<!pt>
// CHECK-SAME: heir.interface = {entry_arg_indices = array<i64>, func_name = "hoist_one_assign", roles = ["server.preprocessing"]}

// CHECK: func.func @hoist_one_assign__preprocessed(%[[ct:.*]]: ![[ct_L1]], %[[arg0:.*]]: !preprocessing.storage<!pt>) -> ![[ct_L1]]
// CHECK-SAME: heir.interface = {func_name = "hoist_one_assign", roles = ["client.preprocessed", "server.evaluate"]}
// CHECK: %[[LOAD:.*]] = preprocessing.load %[[arg0]][] site 0
// CHECK: %[[CT_0:.*]] = ckks.add_plain %ct, %[[LOAD]]
// CHECK: return %[[CT_0]] : ![[ct_L1]]

// CHECK: func.func @hoist_one_assign
// CHECK-SAME: (%[[CT:.*]]: ![[ct_L1]]
// CHECK-NEXT:   %[[STORAGE:.*]] = call @hoist_one_assign__preprocessing()
// CHECK-NEXT:   %[[CALL:.*]] = call @hoist_one_assign__preprocessed(%[[CT]], %[[STORAGE]])
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

func.func @hoist_one_assign(%ct: !ct_L1) -> (!ct_L1) {
  %c1 = arith.constant dense<1.0> : tensor<1024xf32>
  %pt = lwe.rlwe_encode %c1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
  %0 = ckks.add_plain %ct, %pt : (!ct_L1, !pt) -> !ct_L1
  return %0 : !ct_L1
}

// Splitting preserves the logical signature and identity across symbol renames.
// CHECK: func.func @renamed__preprocessing
// CHECK-SAME: heir.interface = {entry_arg_indices = array<i64>, func_name = "logical_entry", roles = ["server.preprocessing"]}
// CHECK: func.func @renamed__preprocessed
// CHECK-SAME: heir.interface = {func_name = "logical_entry", roles = ["client.preprocessed", "server.evaluate"]}
// CHECK: func.func @renamed(
// CHECK-SAME: heir.interface = {extra = "keep", func_name = "logical_entry", input_types = [tensor<16xf32>], result_types = [tensor<16xf32>], roles = ["entry"]}
// CHECK-NOT: "server.evaluate"
// CHECK: return
func.func @renamed(%ct: !ct_L1) -> !ct_L1 attributes {heir.interface = {extra = "keep", func_name = "logical_entry", input_types = [tensor<16xf32>], result_types = [tensor<16xf32>], roles = ["entry", "server.evaluate"]}} {
  %c1 = arith.constant dense<1.0> : tensor<1024xf32>
  %pt = lwe.rlwe_encode %c1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
  %0 = ckks.add_plain %ct, %pt : (!ct_L1, !pt) -> !ct_L1
  return %0 : !ct_L1
}
