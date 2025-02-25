// RUN: heir-opt %s --wrap-generic | FileCheck %s

// CHECK-LABEL: @pointwise
// CHECK-SAME: (%[[ARG0:.*]]: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>>
func.func @pointwise(%arg0: tensor<8xi16> {secret.secret}) -> tensor<8xi16> {
  // CHECK: secret.generic
  return %arg0 : tensor<8xi16>
}

// CHECK-LABEL: @dot_product
// CHECK-SAME: (%[[ARG0:.*]]: !secret.secret<tensor<8xi16>>, %[[ARG1:.*]]: tensor<8xi16>) -> !secret.secret<i16>
func.func @dot_product(%arg0: tensor<8xi16> {secret.secret}, %arg1: tensor<8xi16>) -> i16 {
  // CHECK: secret.generic
  // CHECK:   secret.conceal
  // CHECK:   call
  // CHECK:   secret.reveal
  %pointwise = call @pointwise(%arg0) : (tensor<8xi16>) -> tensor<8xi16>
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %pointwise[%c0] : tensor<8xi16>
  return %0 : i16
}
