// RUN: heir-opt -arith-ext-to-arith --split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_lower_subifge
// CHECK-SAME: (%[[LHS:.*]]: [[TENSOR_TYPE:.*]], %[[RHS:.*]]: [[TENSOR_TYPE]]) -> [[TENSOR_TYPE]] {
func.func @test_lower_subifge(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {

  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[TENSOR_TYPE]]
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[TENSOR_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[LHS]] : tensor<4xi1>, [[TENSOR_TYPE]]
  %res = arith_ext.subifge %lhs, %rhs: tensor<4xi8>
  return %res : tensor<4xi8>
}

// -----

// CHECK-LABEL: @test_lower_normalised
// CHECK-SAME: (%[[ARG:.*]]: [[TENSOR_TYPE:.*]]) -> [[TENSOR_TYPE]] {
func.func @test_lower_normalised(%arg : tensor<4xi8>) -> tensor<4xi8> {
  %res = arith_ext.normalised %arg { q = 17 } : tensor<4xi8>

  // CHECK: return %[[ARG]] : [[TENSOR_TYPE]]
  return %res : tensor<4xi8>
}

// -----

// CHECK-LABEL: @test_lower_normalised_int
// CHECK-SAME: (%[[ARG:.*]]: [[INT_TYPE:.*]]) -> [[INT_TYPE]] {
func.func @test_lower_normalised_int(%arg : i8) -> i8 {
  %res = arith_ext.normalised %arg { q = 17 } : i8

  // CHECK: return %[[ARG]] : i8
  return %res : i8
}

// -----

// CHECK-LABEL: @test_lower_subifge_int
// CHECK-SAME: (%[[LHS:.*]]: [[INT_TYPE:.*]], %[[RHS:.*]]: [[INT_TYPE]]) -> [[INT_TYPE]] {
func.func @test_lower_subifge_int(%lhs : i8, %rhs : i8) -> i8 {

  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[INT_TYPE]]
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[INT_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[LHS]] : [[INT_TYPE]]
  %res = arith_ext.subifge %lhs, %rhs: i8
  return %res : i8
}

// -----

// CHECK-LABEL: @test_lower_barrett_reduce

// CHECK-SAME: (%[[ARG:.*]]: [[TENSOR_TYPE:.*]]) -> [[TENSOR_TYPE]] {
func.func @test_lower_barrett_reduce(%arg : tensor<4xi10>) -> tensor<4xi10> {

  // CHECK: %[[RATIO:.*]] = arith.constant dense<60> : [[INTER_TYPE:.*]]
  // CHECK: %[[BITWIDTH:.*]] = arith.constant dense<10> : [[INTER_TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant dense<17> : [[INTER_TYPE]]

  // CHECK: %[[EXT:.*]] = arith.extui %[[ARG]] : [[TENSOR_TYPE]] to [[INTER_TYPE]]
  // CHECK: %[[MULRATIO:.*]] = arith.muli %[[EXT]], %[[RATIO]] : [[INTER_TYPE]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui %[[MULRATIO]], %[[BITWIDTH]] : [[INTER_TYPE]]
  // CHECK: %[[MULCMOD:.*]] = arith.muli %[[SHIFTED]], %[[CMOD]] : [[INTER_TYPE]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[EXT]], %[[MULCMOD]] : [[INTER_TYPE]]
  // CHECK: %[[RES:.*]] = arith.trunci %[[SUB]] : [[INTER_TYPE]] to [[TENSOR_TYPE]]
  %res = arith_ext.barrett_reduce %arg { modulus = 17 } : tensor<4xi10>

  // CHECK: return %[[RES]] : [[TENSOR_TYPE]]
  return %res : tensor<4xi10>
}

// -----

// CHECK-LABEL: @test_lower_barrett_reduce_int
// CHECK-SAME: (%[[ARG:.*]]: [[INT_TYPE:.*]]) -> [[INT_TYPE]] {
func.func @test_lower_barrett_reduce_int(%arg : i10) -> i10 {

  // CHECK: %[[RATIO:.*]] = arith.constant 60 : [[INTER_TYPE:.*]]
  // CHECK: %[[BITWIDTH:.*]] = arith.constant 10 : [[INTER_TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant 17 : [[INTER_TYPE]]

  // CHECK: %[[EXT:.*]] = arith.extui %[[ARG]] : [[INT_TYPE]] to [[INTER_TYPE]]
  // CHECK: %[[MULRATIO:.*]] = arith.muli %[[EXT]], %[[RATIO]] : [[INTER_TYPE]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui %[[MULRATIO]], %[[BITWIDTH]] : [[INTER_TYPE]]
  // CHECK: %[[MULCMOD:.*]] = arith.muli %[[SHIFTED]], %[[CMOD]] : [[INTER_TYPE]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[EXT]], %[[MULCMOD]] : [[INTER_TYPE]]
  // CHECK: %[[RES:.*]] = arith.trunci %[[SUB]] : [[INTER_TYPE]] to [[INT_TYPE]]
  %res = arith_ext.barrett_reduce %arg { modulus = 17 } : i10

  // CHECK: return %[[RES]] : [[INT_TYPE]]
  return %res : i10
}

// -----
