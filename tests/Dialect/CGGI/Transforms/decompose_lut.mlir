// RUN: heir-opt --cggi-decompose-operations --canonicalize %s | FileCheck %s

#key = #lwe.key<slot_index = 0>
#preserve_overflow = #lwe.preserve_overflow<>
#app_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!ct_ty = !lwe.lwe_ciphertext<application_data = #app_data, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: @lut2
// CHECK-SAME: %[[arg0:.*]]: ![[ct:.*]], %[[arg1:.*]]: ![[ct]]
func.func @lut2(%arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  // CHECK-DAG: %[[const2:.*]] = arith.constant 2 : i3
  // CHECK: %[[mul_b:.*]] = lwe.mul_scalar %[[arg0]], %[[const2]]
  // CHECK: %[[res:.*]] = lwe.add %[[mul_b]], %[[arg1]]
  // CHECK: %[[pbs:.*]] = cggi.programmable_bootstrap %[[res]] {lookup_table = 8 : ui8}
  // CHECK: return %[[pbs]]
  %r1 = cggi.lut2 %arg0, %arg1 {lookup_table = 8 : ui8} : !ct_ty
  return %r1 : !ct_ty
}

// CHECK: @lut3
// CHECK-SAME: %[[arg0:.*]]: ![[ct:.*]], %[[arg1:.*]]: ![[ct]], %[[arg2:.*]]: ![[ct]]
func.func @lut3(%arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty) -> !ct_ty {
  // CHECK-DAG: %[[const4:.*]] = arith.constant -4 : i3
  // CHECK-DAG: %[[const2:.*]] = arith.constant 2 : i3
  // CHECK: %[[mul_c:.*]] = lwe.mul_scalar %[[arg0]], %[[const4]]
  // CHECK: %[[mul_b:.*]] = lwe.mul_scalar %[[arg1]], %[[const2]]
  // CHECK: %[[add_cb:.*]] = lwe.add %[[mul_c]], %[[mul_b]]
  // CHECK: %[[res:.*]] = lwe.add %[[add_cb]], %[[arg2]]
  // CHECK: %[[pbs:.*]] = cggi.programmable_bootstrap %[[res]] {lookup_table = 8 : ui8}
  // CHECK: return %[[pbs]]
  %r1 = cggi.lut3 %arg0, %arg1, %arg2 {lookup_table = 8 : ui8} : !ct_ty
  return %r1 : !ct_ty
}

// CHECK: @lut_lincomb
// CHECK-SAME: %[[arg0:.*]]: ![[ct:.*]], %[[arg1:.*]]: ![[ct]]
func.func @lut_lincomb(%arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  // CHECK-DAG: %[[const3:.*]] = arith.constant 3 : i3
  // CHECK-DAG: %[[const6:.*]] = arith.constant -2 : i3
  // CHECK: %[[mul_c:.*]] = lwe.mul_scalar %[[arg0]], %[[const3]]
  // CHECK: %[[mul_b:.*]] = lwe.mul_scalar %[[arg1]], %[[const6]]
  // CHECK: %[[add_cb:.*]] = lwe.add %[[mul_c]], %[[mul_b]]
  // CHECK: %[[pbs:.*]] = cggi.programmable_bootstrap %[[add_cb]] {lookup_table = 68 : index}
  // CHECK: return %[[pbs]]
  %r1 = cggi.lut_lincomb %arg0, %arg1 {coefficients = array<i32: 3, 6>, lookup_table = 68 : index} : !ct_ty
  return %r1 : !ct_ty
}

#pspace4 = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i4, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace4 = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!ct_ty4 = !lwe.lwe_ciphertext<application_data = #app_data, plaintext_space = #pspace4, ciphertext_space = #cspace4, key = #key>

// CHECK: @lut_bitwidth_4
// CHECK-SAME: %[[arg0:.*]]: ![[ct:.*]], %[[arg1:.*]]: ![[ct]]
func.func @lut_bitwidth_4(%arg0: !ct_ty4, %arg1: !ct_ty4) -> !ct_ty4 {
  // CHECK-DAG: %[[const3:.*]] = arith.constant 3 : i4
  // CHECK-DAG: %[[const6:.*]] = arith.constant 6 : i4
  // CHECK: %[[mul_c:.*]] = lwe.mul_scalar %[[arg0]], %[[const3]]
  // CHECK: %[[mul_b:.*]] = lwe.mul_scalar %[[arg1]], %[[const6]]
  // CHECK: %[[add_cb:.*]] = lwe.add %[[mul_c]], %[[mul_b]]
  // CHECK: %[[pbs:.*]] = cggi.programmable_bootstrap %[[add_cb]] {lookup_table = 68 : index}
  // CHECK: return %[[pbs]]
  %r1 = cggi.lut_lincomb %arg0, %arg1 {coefficients = array<i32: 3, 6>, lookup_table = 68 : index} : !ct_ty4
  return %r1 : !ct_ty4
}
