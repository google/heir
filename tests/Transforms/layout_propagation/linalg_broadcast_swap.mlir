// RUN: heir-opt --layout-propagation --split-input-file %s | FileCheck %s
// This tests the broadcast swap on a linalg broadcast

// CHECK-DAG: #layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 1023 }">
module {
  // CHECK: func @main
  func.func @main(%arg0: !secret.secret<tensor<8xf32>>) -> (!secret.secret<tensor<8xf32>>) {
    %cst = arith.constant dense<0.0> : tensor<f32>
    %cst_0 = arith.constant dense<2.0> : tensor<8xf32>
    %0 = tensor.empty() {secret.public} : tensor<8xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>):
      // CHECK-DAG linalg.reduce ins(%input0 : tensor<8xf32>) outs(%cst : tensor<f32>) dimensions = [0] {tensor_ext.layout = [[#layout]]}
      %reduced = linalg.reduce ins(%input0 : tensor<8xf32>) outs(%cst : tensor<f32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %31 = arith.addf %in, %init {secret.public} : f32
          linalg.yield %31 {secret.public} : f32
        }
      // CHECK-DAG linalg.broadcast ins(%reduced : tensor<f32>) outs(%0 : tensor<8xf32>) dimensions = [0] {tensor_ext.layout = [[#layout]]}
      %broadcasted = linalg.broadcast ins(%reduced : tensor<f32>) outs(%0 : tensor<8xf32>) dimensions = [0]
      // CHECK-DAG arith.mulf %broadcasted, %cst_0 : tensor<8xf32> {tensor_ext.layout = [[#layout]]}
      %2 = arith.mulf %broadcasted, %cst_0 : tensor<8xf32>
      secret.yield %2: tensor<8xf32>
    } -> !secret.secret<tensor<8xf32>>
  return %1 : !secret.secret<tensor<8xf32>>
  }
}
