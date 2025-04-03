// RUN: heir-opt --cggi-to-tfhe-rust -cse %s | FileCheck %s

#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 3>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>
// CHECK: @binary_gates
// CHECK-SAME: %[[sks:.*]]: [[sks_ty:!tfhe_rust.server_key]], %[[arg1:.*]]: [[ct_ty:!tfhe_rust.eui3]], %[[arg2:.*]]: [[ct_ty]]
func.func @binary_gates(%arg1: !ct_ty, %arg2: !ct_ty) -> (!ct_ty) {
  // CHECK: %[[v0:.*]] = tfhe_rust.generate_lookup_table %[[sks]] {truthTable = 8 : ui4}
  // CHECK: %[[v1:.*]] = tfhe_rust.scalar_left_shift %[[sks]], %[[arg2]] {shiftAmount = 1 : index}
  // CHECK: %[[v2:.*]] = tfhe_rust.add %[[sks]], %[[v1]], %[[arg1]]
  // CHECK: %[[v3:.*]] = tfhe_rust.apply_lookup_table %[[sks]], %[[v2]], %[[v0]]
  %0 = cggi.and %arg1, %arg2 : !ct_ty

  // CHECK: %[[v4:.*]] = tfhe_rust.generate_lookup_table %[[sks]] {truthTable = 14 : ui4}
  // CHECK: %[[v5:.*]] = tfhe_rust.apply_lookup_table %[[sks]], %[[v2]], %[[v4]]
  // (reuses shifted inputs from the AND)
  %1 = cggi.or %arg1, %arg2 : !ct_ty

  // CHECK: %[[notConst:.*]] = arith.constant 1 : i3
  // CHECK: %[[v6:.*]] = tfhe_rust.create_trivial %[[sks]], %[[notConst]]
  // CHECK: %[[v7:.*]] = tfhe_rust.sub %[[sks]], %[[v6]], %[[v5]]
  %2 = cggi.not %1 : !ct_ty

  // CHECK: %[[v8:.*]] = tfhe_rust.generate_lookup_table %[[sks]] {truthTable = 6 : ui4}
  // CHECK: %[[v9:.*]] = tfhe_rust.scalar_left_shift %[[sks]], %[[v3]] {shiftAmount = 1 : index}
  // CHECK: %[[v10:.*]] = tfhe_rust.add %[[sks]], %[[v9]], %[[v7]]
  // CHECK: %[[v11:.*]] = tfhe_rust.apply_lookup_table %[[sks]], %[[v10]], %[[v8]]
  %3 = cggi.xor %2, %0 : !ct_ty

  // CHECK: return %[[v11]]
  return %3 : !ct_ty
}
