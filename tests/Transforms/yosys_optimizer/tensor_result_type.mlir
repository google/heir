// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

// This test ensures that the memref allocation occurring within the generic op
// has a shape extending the result shape. In this case the allocation is
// tensor<1x1x8xi1> to match the result tensor<1x1xi8>, instead of the flattened
// wire representation tensor<8xi1>.

module attributes {tf_saved_model.semantics} {
  // CHECK: @main(%[[arg0:.*]]: [[SECRETI8:!secret.secret<i8>]],
  // CHECK-SAME:  %[[arg1:.*]]: [[SECRET1x1xi8:!secret.secret<tensor<1x1xi8>>]])
  // CHECK-SAME: -> [[SECRET1x1xi8:!secret.secret<tensor<1x1xi8>>]] {
  // CHECK-NEXT: %[[v0:.*]] = secret.cast %[[arg0]] : [[SECRETI8]] to [[SECRET8xi1:!secret.secret<tensor<8xi1>>]]
  // CHECK-NEXT: secret.generic(%[[v0]]: [[SECRET8xi1]])
  // CHECK:         %[[from_elements:.*]] = tensor.from_elements
  // CHECK-SAME:        [[TENSOR8xi1:tensor<8xi1>]]
  // CHECK:         secret.yield %[[from_elements]] : [[TENSOR8xi1]]
  func.func @main(%arg0: !secret.secret<i8>, %out: !secret.secret<tensor<1x1xi8>>) -> !secret.secret<tensor<1x1xi8>> {
    %c22 = arith.constant 22 : i8
    %c0 = arith.constant 0 : index
    %0 = secret.generic(%arg0 : !secret.secret<i8>, %out : !secret.secret<tensor<1x1xi8>>) {
    ^bb0(%arg1: i8, %arg2: tensor<1x1xi8>):
      %1 = arith.addi %arg1, %c22 : i8
      %inserted = tensor.insert %1 into %arg2[%c0, %c0] : tensor<1x1xi8>
      secret.yield %inserted : tensor<1x1xi8>
    } -> !secret.secret<tensor<1x1xi8>>
    return %0 : !secret.secret<tensor<1x1xi8>>
  }
}
