// RUN: heir-opt %s --wrap-generic | FileCheck %s

// CHECK-LABEL: @dot_product
// CHECK-SAME: (%[[ARG0:.*]]: !secret.secret<tensor<8xi16>>, %[[ARG1:.*]]: tensor<8xi16>) -> !secret.secret<i16>
func.func @dot_product(%arg0: tensor<8xi16> {secret.secret}, %arg1: tensor<8xi16>) -> i16 {
  // CHECK: secret.generic
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi16>
  return %0 : i16
}

// CHECK-LABEL: @test_dot_product
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xi16>, %[[ARG1:.*]]: tensor<8xi16>
func.func @test_dot_product(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>) {
  // CHECK: %[[v0:.*]] = secret.conceal %[[ARG0]]
  // CHECK: %[[v1:.*]] = call @dot_product(%[[v0]], %[[ARG1]])
  // CHECK: %[[v2:.*]] = secret.reveal %[[v1]]
  // CHECK: return
  %res = func.call @dot_product(%arg0, %arg1) : (tensor<8xi16>, tensor<8xi16>) -> i16
  return
}
