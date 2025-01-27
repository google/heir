// RUN: heir-opt --arith-to-cggi --split-input-file %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, %[[RHS:.*]]: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> [[T:.*]] {
func.func @test_lower_add(%lhs : i32, %rhs : i32) -> i32 {
  // CHECK: %[[ADD:.*]] = cggi.add %[[LHS]], %[[RHS]] : ([[T]], [[T]]) -> [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.addi %lhs, %rhs : i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_cte_add
// CHECK-SAME: (%[[LHS:.*]]: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> [[T:.*]] {
func.func @test_lower_cte_add(%in : i32) -> i32 {
  // CHECK: %[[CTE:.*]] = arith.constant 7 : i32
  // CHECK: %[[ADD:.*]] = cggi.add %[[LHS]], %[[CTE]] : ([[T]], i32) -> [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %c7 = arith.constant 7 : i32
  %res = arith.addi %in, %c7 : i32
  return %res : i32
}


// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding1>, %[[RHS:.*]]: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding1>) -> [[T:.*]] {
func.func @test_lower_sub(%lhs : i16, %rhs : i16) -> i16 {
  // CHECK: %[[ADD:.*]] = cggi.sub %[[LHS]], %[[RHS]] : ([[T]], [[T]]) -> [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.subi %lhs, %rhs : i16
  return %res : i16
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding2>, %[[RHS:.*]]: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding2>) -> [[T:.*]] {
func.func @test_lower_mul(%lhs : i8, %rhs : i8) -> i8 {
  // CHECK: %[[ADD:.*]] = cggi.mul %[[LHS]], %[[RHS]] : ([[T]], [[T]]) -> [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.muli %lhs, %rhs : i8
  return %res : i8
}

// CHECK-LABEL: @test_affine
// CHECK-SAME: (%[[ARG:.*]]: memref<1x1x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>) -> [[T:.*]] {
func.func @test_affine(%arg0: memref<1x1xi32>) -> memref<1x1xi32> {
  // CHECK: return %[[ADD:.*]] : [[T]]
  %c429_i32 = arith.constant 429 : i32
  %c33_i8 = arith.constant 33 : i32
  %0 = affine.load %arg0[0, 0] : memref<1x1xi32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
  %25 = arith.muli %0, %c33_i8 : i32
  %26 = arith.addi %c429_i32, %25 : i32
  %c2 = arith.constant 2 : i32
  %27 = arith.shrui %26, %c2 : i32
  affine.store %26, %alloc[0, 0] : memref<1x1xi32>
  return %alloc : memref<1x1xi32>
}
