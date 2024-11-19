// RUN: heir-opt --arith-to-mod-arith --split-input-file %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: !Z2147483647_i32_, %[[RHS:.*]]: !Z2147483647_i32_) -> [[T:.*]] {
func.func @test_lower_add(%lhs : i32, %rhs : i32) -> i32 {
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.addi %lhs, %rhs : i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x!Z2147483647_i32_>, %[[RHS:.*]]: tensor<4x!Z2147483647_i32_>) -> [[T:.*]] {
func.func @test_lower_add_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.addi %lhs, %rhs : tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[ADD:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.subi %lhs, %rhs : tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : i32, %rhs : i32) -> i32 {
  // CHECK: %[[ADD:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.subi %lhs, %rhs : i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul(%lhs : i32, %rhs : i32) -> i32 {
  // CHECK: %[[ADD:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.muli %lhs, %rhs : i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[ADD:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.muli %lhs, %rhs : tensor<4xi32>
  return %res : tensor<4xi32>
}
