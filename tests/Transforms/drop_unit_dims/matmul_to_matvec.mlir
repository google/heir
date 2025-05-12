// RUN: heir-opt %s --drop-unit-dims | FileCheck %s

// CHECK: collapse_matmul_rhs
// CHECK-NOT: linalg.matmul
// CHECK: tensor.collapse_shape
// CHECK: linalg.matvec
func.func @collapse_matmul_rhs(%vec : !secret.secret<tensor<4x1xi16>>) -> !secret.secret<tensor<4x1xi16>> {
  %matrix = arith.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<4x4xi16>
  %bias = arith.constant dense<[[17], [18], [19], [20]]> : tensor<4x1xi16>
  %out = secret.generic (%vec : !secret.secret<tensor<4x1xi16>>) {
  ^body(%pt_vec: tensor<4x1xi16>):
    %0 = linalg.matmul ins(%matrix, %pt_vec : tensor<4x4xi16>, tensor<4x1xi16>) outs(%bias : tensor<4x1xi16>) -> tensor<4x1xi16>
    secret.yield %0 : tensor<4x1xi16>
  } -> !secret.secret<tensor<4x1xi16>>
  return %out : !secret.secret<tensor<4x1xi16>>
}

// CHECK: collapse_matmul_lhs
// CHECK-NOT: linalg.matmul
// CHECK: tensor.collapse_shape
// CHECK: linalg.vecmat
func.func @collapse_matmul_lhs(%vec : !secret.secret<tensor<1x4xi16>>) -> !secret.secret<tensor<1x4xi16>> {
  %matrix = arith.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<4x4xi16>
  %bias = arith.constant dense<[[17, 18, 19, 20]]> : tensor<1x4xi16>
  %out = secret.generic (%vec : !secret.secret<tensor<1x4xi16>>) {
  ^body(%pt_vec: tensor<1x4xi16>):
    %0 = linalg.matmul ins(%pt_vec, %matrix : tensor<1x4xi16>, tensor<4x4xi16>) outs(%bias : tensor<1x4xi16>) -> tensor<1x4xi16>
    secret.yield %0 : tensor<1x4xi16>
  } -> !secret.secret<tensor<1x4xi16>>
  return %out : !secret.secret<tensor<1x4xi16>>
}
