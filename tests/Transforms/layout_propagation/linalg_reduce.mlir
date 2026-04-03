// RUN: heir-opt --layout-propagation --fold-convert-layout-into-assign-layout %s | FileCheck %s

// CHECK: @main
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<8xf32>> {tensor_ext.layout = [[rm_layout:.*]]}) ->
module {
  func.func @main(%arg0: !secret.secret<tensor<8xf32>>, %arg1: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<f32>> {
    // CHECK-DAG: %[[cst:.*]] = arith.constant
    // CHECK-DAG: tensor_ext.assign_layout %[[cst]]
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>>, %arg1: !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>, %input1: tensor<8xf32>):
      %1 = arith.mulf %input0, %input1 : tensor<8xf32>
      %reduced = linalg.reduce ins(%1 : tensor<8xf32>) outs(%cst : tensor<f32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %2 = arith.addf %in, %init : f32
          linalg.yield %2 : f32
        }
      secret.yield %reduced : tensor<f32>
    } -> !secret.secret<tensor<f32>>
    return %0 : !secret.secret<tensor<f32>>
  }
}
