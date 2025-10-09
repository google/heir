// RUN: heir-opt --cggi-decompose-operations --cggi-to-tfhe-rust -cse %s | FileCheck %s --check-prefixes=CHECK,CHECK-COMMON
// RUN: heir-opt --cggi-to-tfhe-rust -cse %s | FileCheck %s --check-prefixes=CHECK-GATES,CHECK-COMMON

#key = #lwe.key<slot_index = 0>
#preserve_overflow = #lwe.preserve_overflow<>
#app_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!pt_ty = !lwe.lwe_plaintext<application_data = #app_data, plaintext_space = #pspace>
!ct_ty = !lwe.lwe_ciphertext<application_data = #app_data, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK-COMMON: @binary_gates
// CHECK-COMMON-SAME: %[[sks:.*]]: [[sks_ty:!tfhe_rust.server_key]], %[[arg1:.*]]: [[ct_ty:!tfhe_rust.eui3]], %[[arg2:.*]]: [[ct_ty]]
func.func @binary_gates(%arg1: !ct_ty, %arg2: !ct_ty) -> (!ct_ty) {
  // CHECK-DAG: %[[v0:.*]] = tfhe_rust.generate_lookup_table %[[sks]] {truthTable = 8 : ui4}
  // CHECK-DAG: %[[v1:.*]] = tfhe_rust.scalar_left_shift %[[sks]], %[[arg2]] {shiftAmount = 1 : index}
  // CHECK-DAG: %[[v2:.*]] = tfhe_rust.add %[[sks]], %[[v1]], %[[arg1]]
  // CHECK: %[[v3:.*]] = tfhe_rust.apply_lookup_table %[[sks]], %[[v2]], %[[v0]]
  // CHECK-GATES: %[[v0:.*]] = tfhe_rust.bitand %[[sks]], %[[arg1]], %[[arg2]]
  %0 = cggi.and %arg1, %arg2 : !ct_ty

  // CHECK: %[[v4:.*]] = tfhe_rust.generate_lookup_table %[[sks]] {truthTable = 14 : ui4}
  // CHECK: %[[v5:.*]] = tfhe_rust.apply_lookup_table %[[sks]], %[[v2]], %[[v4]]
  // (reuses shifted inputs from the AND)
  // CHECK-GATES: %[[v1:.*]] = tfhe_rust.bitor %[[sks]], %[[arg1]], %[[arg2]]
  %1 = cggi.or %arg1, %arg2 : !ct_ty

  // CHECK-COMMON: %[[v6:.*]] = tfhe_rust.create_trivial %[[sks]] {valueAttr = 1 : i3}
  // CHECK: %[[v7:.*]] = tfhe_rust.sub %[[sks]], %[[v6]], %[[v5]]
  // CHECK-GATES: %[[v2:.*]] = tfhe_rust.sub %[[sks]], %[[v6]], %[[v1]]
  %2 = cggi.not %1 : !ct_ty

  // CHECK: %[[v9:.*]] = tfhe_rust.scalar_left_shift %[[sks]], %[[v3]] {shiftAmount = 1 : index}
  // CHECK: %[[v10:.*]] = tfhe_rust.add %[[sks]], %[[v9]], %[[v7]]
  // CHECK: %[[v8:.*]] = tfhe_rust.generate_lookup_table %[[sks]] {truthTable = 6 : ui4}
  // CHECK: %[[v11:.*]] = tfhe_rust.apply_lookup_table %[[sks]], %[[v10]], %[[v8]]
  // CHECK-GATES: %[[v3:.*]] = tfhe_rust.bitxor %[[sks]], %[[v2]], %[[v0]]
  %3 = cggi.xor %2, %0 : !ct_ty

  // CHECK: return %[[v11]]
  // CHECK-GATES: return %[[v3]]
  return %3 : !ct_ty
}
