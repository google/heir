// RUN: heir-opt --cggi-to-openfhe %s | FileCheck %s

#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<1 + x**1024>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 2>
!ct_ty = !lwe.lwe_ciphertext<application_data = <message_type = i3, overflow = #preserve_overflow>, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// Test LutLinComb operation conversion
// CHECK: @test_lut_lincomb
// CHECK-SAME: (%[[CTX:[^:]*]]: !ctx
// CHECK-SAME: %[[ARG0:[^:]*]]: !ct_
// CHECK-SAME: %[[ARG1:[^:]*]]: !ct_
// CHECK-SAME: %[[ARG2:[^:]*]]: !ct_
// CHECK-SAME: %[[ARG3:[^:]*]]: !ct_
func.func @test_lut_lincomb(%arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty) -> !ct_ty {
  // CHECK: %[[SCHEME:[^=]*]] = openfhe.get_lwe_scheme %[[CTX]]
  // CHECK: %[[C1:[^=]*]] = arith.constant 1 : i64
  // CHECK: %[[MUL0:[^=]*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG0]], %[[C1]]
  // CHECK: %[[C2:[^=]*]] = arith.constant 2 : i64
  // CHECK: %[[MUL1:[^=]*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG1]], %[[C2]]
  // CHECK: %[[C3:[^=]*]] = arith.constant 3 : i64
  // CHECK: %[[MUL2:[^=]*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG2]], %[[C3]]
  // CHECK: %[[C4:[^=]*]] = arith.constant 2 : i64
  // CHECK: %[[MUL3:[^=]*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG3]], %[[C4]]
  // CHECK: %[[ADD0:[^=]*]] = openfhe.lwe_add %[[SCHEME]], %[[MUL0]], %[[MUL1]]
  // CHECK: %[[ADD1:[^=]*]] = openfhe.lwe_add %[[SCHEME]], %[[ADD0]], %[[MUL2]]
  // CHECK: %[[ADD2:[^=]*]] = openfhe.lwe_add %[[SCHEME]], %[[ADD1]], %[[MUL3]]
  // CHECK: %[[LUT:[^=]*]] = openfhe.make_lut %[[CTX]] {values = array<i32: 2, 6>}
  // CHECK: %[[RESULT:[^=]*]] = openfhe.eval_func %[[CTX]], %[[LUT]], %[[ADD2]]
  // CHECK: return %[[RESULT]]
  %0 = cggi.lut_lincomb %arg0, %arg1, %arg2, %arg3 {coefficients = array<i32: 1, 2, 3, 2>, lookup_table = 68 : index} : !ct_ty
  return %0 : !ct_ty
}
