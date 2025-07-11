// RUN: heir-opt --arith-to-cggi --split-input-file %s | FileCheck %s --enable-var-scope

// CHECK: ![[ct_32:.*]] = !lwe.new_lwe_ciphertext<application_data = <message_type = i32,
// CHECK: ![[ct_16:.*]] = !lwe.new_lwe_ciphertext<application_data = <message_type = i16,
// CHECK: ![[ct_8:.*]] = !lwe.new_lwe_ciphertext<application_data = <message_type = i8,

// CHECK: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: ![[ct_32]], %[[RHS:.*]]: ![[ct_32]]) -> ![[ct_32]] {
func.func @test_lower_add(%lhs : i32, %rhs : i32) -> i32 {
  // CHECK: %[[ADD:.*]] = cggi.add %[[LHS]], %[[RHS]] : (![[ct_32]], ![[ct_32]]) -> ![[ct_32]]
  // CHECK: return %[[ADD:.*]] : ![[ct_32]]
  %res = arith.addi %lhs, %rhs : i32
  return %res : i32
}

// CHECK: @test_lower_cte_add
// CHECK-SAME: (%[[LHS:.*]]: ![[ct_32]]) -> ![[ct_32]] {
func.func @test_lower_cte_add(%in : i32) -> i32 {
  // CHECK: %[[CTE:.*]] = arith.constant 7 : i32
  // CHECK: %[[ADD:.*]] = cggi.add %[[LHS]], %[[CTE]] : (![[ct_32]], i32) -> ![[ct_32]]
  // CHECK: return %[[ADD:.*]] : ![[ct_32]]
  %c7 = arith.constant 7 : i32
  %res = arith.addi %in, %c7 : i32
  return %res : i32
}


// CHECK: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: ![[ct_16]], %[[RHS:.*]]: ![[ct_16]]) -> ![[ct_16]] {
func.func @test_lower_sub(%lhs : i16, %rhs : i16) -> i16 {
  // CHECK: %[[ADD:.*]] = cggi.sub %[[LHS]], %[[RHS]] : (![[ct_16]], ![[ct_16]]) -> ![[ct_16]]
  // CHECK: return %[[ADD:.*]] : ![[ct_16]]
  %res = arith.subi %lhs, %rhs : i16
  return %res : i16
}

// CHECK: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: ![[ct_8]], %[[RHS:.*]]: ![[ct_8]]) -> ![[ct_8]] {
func.func @test_lower_mul(%lhs : i8, %rhs : i8) -> i8 {
  // CHECK: %[[ADD:.*]] = cggi.mul %[[LHS]], %[[RHS]] : (![[ct_8]], ![[ct_8]]) -> ![[ct_8]]
  // CHECK: return %[[ADD:.*]] : ![[ct_8]]
  %res = arith.muli %lhs, %rhs : i8
  return %res : i8
}

// CHECK: @test_affine
// CHECK-SAME: (%[[ARG:.*]]: memref<1x1x![[ct_32]]>) -> memref<1x1x![[ct_32]]> {
func.func @test_affine(%arg0: memref<1x1xi32>) -> memref<1x1xi32> {
  // CHECK: return %[[ADD:.*]] : memref<1x1x![[ct_32]]>
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
