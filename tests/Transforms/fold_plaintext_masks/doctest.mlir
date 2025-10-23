// RUN: heir-opt %s --fold-plaintext-masks | FileCheck %s

// CHECK: @example
// CHECK-COUNT-1: arith.muli
func.func @example(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  %cst_1 = arith.constant dense<[1, 1, 1, 0, 0, 0]> : tensor<6xi32>
  %cst_2 = arith.constant dense<[0, 1, 1, 1, 1, 1]> : tensor<6xi32>
  %1 = arith.muli %arg0, %cst_1 : tensor<6xi32>
  %2 = arith.muli %1, %cst_2 : tensor<6xi32>
  return %2 : tensor<6xi32>
}
