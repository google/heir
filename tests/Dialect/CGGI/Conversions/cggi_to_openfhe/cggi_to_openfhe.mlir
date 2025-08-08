// RUN: heir-opt --cggi-to-openfhe %s | FileCheck %s

#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<1 + x**1024>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 2>
!ct_ty = !lwe.lwe_ciphertext<application_data = <message_type = i3, overflow = #preserve_overflow>, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// Test LutLinComb operation conversion
// CHECK: @test_lut_lincomb
// CHECK-SAME: (%[[CTX:.*]]: !ctx, %[[ARG0:.*]]: !ct_, %[[ARG1:.*]]: !ct_, %[[ARG2:.*]]: !ct_, %[[ARG3:.*]]: !ct_)
func.func @test_lut_lincomb(%arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty) -> !ct_ty {
  // CHECK: %[[SCHEME:.*]] = openfhe.get_lwe_scheme %[[CTX]]
  // CHECK: %[[C1:.*]] = arith.constant 1 : i64
  // CHECK: %[[MUL0:.*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG0]], %[[C1]]
  // CHECK: %[[C2:.*]] = arith.constant 2 : i64
  // CHECK: %[[MUL1:.*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG1]], %[[C2]]
  // CHECK: %[[C3:.*]] = arith.constant 3 : i64
  // CHECK: %[[MUL2:.*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG2]], %[[C3]]
  // CHECK: %[[C4:.*]] = arith.constant 2 : i64
  // CHECK: %[[MUL3:.*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG3]], %[[C4]]
  // CHECK: %[[ADD0:.*]] = openfhe.lwe_add %[[SCHEME]], %[[MUL0]], %[[MUL1]]
  // CHECK: %[[ADD1:.*]] = openfhe.lwe_add %[[SCHEME]], %[[ADD0]], %[[MUL2]]
  // CHECK: %[[ADD2:.*]] = openfhe.lwe_add %[[SCHEME]], %[[ADD1]], %[[MUL3]]
  // CHECK: %[[LUT:.*]] = openfhe.make_lut %[[CTX]] {values = array<i32: 2, 6>}
  // CHECK: %[[RESULT:.*]] = openfhe.eval_func %[[CTX]], %[[LUT]], %[[ADD2]]
  // CHECK: return %[[RESULT]]
  %0 = cggi.lut_lincomb %arg0, %arg1, %arg2, %arg3 {coefficients = array<i32: 1, 2, 3, 2>, lookup_table = 68 : index} : !ct_ty
  return %0 : !ct_ty
}

// Test single input LutLinComb
// CHECK: @test_lut_lincomb_single
// CHECK-SAME: (%[[CTX:.*]]: !ctx, %[[ARG0:.*]]: !ct_
func.func @test_lut_lincomb_single(%arg0: !ct_ty) -> !ct_ty {
  // CHECK: %[[SCHEME:.*]] = openfhe.get_lwe_scheme %[[CTX]]
  // CHECK: %[[C3:.*]] = arith.constant 3 : i64
  // CHECK: %[[MUL:.*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG0]], %[[C3]]
  // CHECK: %[[LUT:.*]] = openfhe.make_lut %[[CTX]] {values = array<i32: 3>}
  // CHECK: %[[RESULT:.*]] = openfhe.eval_func %[[CTX]], %[[LUT]], %[[MUL]]
  // CHECK: return %[[RESULT]]
  %0 = cggi.lut_lincomb %arg0 {coefficients = array<i32: 3>, lookup_table = 8 : index} : !ct_ty
  return %0 : !ct_ty
}

// Test two input LutLinComb
// CHECK: @test_lut_lincomb_two
// CHECK-SAME: (%[[CTX:.*]]: !ctx, %[[ARG0:.*]]: !ct_, %[[ARG1:.*]]: !ct_)
func.func @test_lut_lincomb_two(%arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  // CHECK: %[[SCHEME:.*]] = openfhe.get_lwe_scheme %[[CTX]]
  // CHECK: %[[C1:.*]] = arith.constant 1 : i64
  // CHECK: %[[MUL0:.*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG0]], %[[C1]]
  // CHECK: %[[C1_2:.*]] = arith.constant 1 : i64
  // CHECK: %[[MUL1:.*]] = openfhe.lwe_mul_const %[[SCHEME]], %[[ARG1]], %[[C1_2]]
  // CHECK: %[[ADD:.*]] = openfhe.lwe_add %[[SCHEME]], %[[MUL0]], %[[MUL1]]
  // CHECK: %[[LUT:.*]] = openfhe.make_lut %[[CTX]] {values = array<i32: 0, 1, 2, 3>}
  // CHECK: %[[RESULT:.*]] = openfhe.eval_func %[[CTX]], %[[LUT]], %[[ADD]]
  // CHECK: return %[[RESULT]]
  %0 = cggi.lut_lincomb %arg0, %arg1 {coefficients = array<i32: 1, 1>, lookup_table = 15 : index} : !ct_ty
  return %0 : !ct_ty
}

// Test function calls with crypto context propagation
// CHECK: @test_function_call
// CHECK-SAME: (%[[CTX:.*]]: !ctx, %[[ARG0:.*]]: !ct_)
func.func @test_function_call(%arg0: !ct_ty) -> !ct_ty {
  // Add a CGGI op to ensure this function gets a crypto context
  %temp = cggi.lut_lincomb %arg0 {coefficients = array<i32: 1>, lookup_table = 1 : index} : !ct_ty
  // CHECK: call @helper_function
  %0 = func.call @helper_function(%temp) : (!ct_ty) -> !ct_ty
  return %0 : !ct_ty
}

// The helper_function is correctly converted with crypto context
func.func @helper_function(%arg0: !ct_ty) -> !ct_ty {
  %0 = cggi.lut_lincomb %arg0 {coefficients = array<i32: 2>, lookup_table = 3 : index} : !ct_ty
  return %0 : !ct_ty
}

// Test multiple LutLinComb operations in sequence
// CHECK: @test_multiple_luts
// CHECK-SAME: (%[[CTX:.*]]: !ctx, %[[ARG0:.*]]: !ct_, %[[ARG1:.*]]: !ct_)
func.func @test_multiple_luts(%arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  // First LUT operation
  // CHECK: %[[SCHEME:.*]] = openfhe.get_lwe_scheme %[[CTX]]
  %0 = cggi.lut_lincomb %arg0 {coefficients = array<i32: 1>, lookup_table = 2 : index} : !ct_ty

  // Second LUT operation using result of first
  %1 = cggi.lut_lincomb %0, %arg1 {coefficients = array<i32: 1, 2>, lookup_table = 7 : index} : !ct_ty
  return %1 : !ct_ty
}

// Test tensor operations with LutLinComb
// CHECK: @test_tensor_lut
// CHECK-SAME: (%[[CTX:.*]]: !ctx, %[[ARG0:.*]]: tensor<2x!ct_
func.func @test_tensor_lut(%arg0: tensor<2x!ct_ty>) -> tensor<2x!ct_ty> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = tensor.extract %arg0[%c0] : tensor<2x!ct_ty>
  %1 = tensor.extract %arg0[%c1] : tensor<2x!ct_ty>

  %2 = cggi.lut_lincomb %0, %1 {coefficients = array<i32: 1, 1>, lookup_table = 3 : index} : !ct_ty
  %3 = cggi.lut_lincomb %1 {coefficients = array<i32: 2>, lookup_table = 1 : index} : !ct_ty

  %4 = tensor.from_elements %2, %3 : tensor<2x!ct_ty>
  return %4 : tensor<2x!ct_ty>
}

// Test with different lookup table values
// CHECK: @test_various_luts
// CHECK-SAME: (%[[CTX:.*]]: !ctx, %[[ARG0:.*]]: !ct_
func.func @test_various_luts(%arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty) -> (!ct_ty, !ct_ty, !ct_ty) {
  // LUT with all bits set
  %0 = cggi.lut_lincomb %arg0 {coefficients = array<i32: 1>, lookup_table = 255 : index} : !ct_ty

  // LUT with alternating bits
  %1 = cggi.lut_lincomb %arg1 {coefficients = array<i32: 1>, lookup_table = 170 : index} : !ct_ty

  // LUT for XOR operation
  %2 = cggi.lut_lincomb %arg0, %arg1 {coefficients = array<i32: 1, 2>, lookup_table = 6 : index} : !ct_ty

  return %0, %1, %2 : !ct_ty, !ct_ty, !ct_ty
}

// Test memref operations with LutLinComb
// CHECK: @test_memref_lut
// CHECK-SAME: (%[[CTX:.*]]: !ctx, %[[ARG0:.*]]: memref<4x!ct_
func.func @test_memref_lut(%arg0: memref<4x!ct_ty>) -> !ct_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<4x!ct_ty>
  %1 = memref.load %arg0[%c1] : memref<4x!ct_ty>
  %2 = memref.load %arg0[%c2] : memref<4x!ct_ty>
  %3 = memref.load %arg0[%c3] : memref<4x!ct_ty>

  %4 = cggi.lut_lincomb %0, %1, %2, %3 {coefficients = array<i32: 1, 1, 1, 1>, lookup_table = 128 : index} : !ct_ty
  return %4 : !ct_ty
}

// Test empty function that should not get crypto context
// CHECK: @internal_generic_empty_function
// CHECK-NOT: !ctx
func.func @internal_generic_empty_function(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  return %0 : i32
}

// Test internal generic function that should not get crypto context
// CHECK: @internal_generic_function
// CHECK-NOT: !ctx
func.func @internal_generic_function(%arg0: i32) -> i32 {
  return %arg0 : i32
}
