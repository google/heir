// RUN: heir-opt --layout-propagation --fold-convert-layout-into-assign-layout %s | FileCheck %s

// CHECK: @broadcast_back
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<8xf32>> {{{.*}}tensor_ext.layout = [[layout_A:.*]]}) ->
module {
  func.func @broadcast_back(%arg0: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>):
      // CHECK: linalg.reduce
      %reduced = linalg.reduce ins(%input0 : tensor<8xf32>) outs(%cst : tensor<f32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %2 = arith.addf %in, %init : f32
          linalg.yield %2 : f32
        }
      // CHECK: linalg.broadcast
      // CHECK-NOT: tensor_ext.convert_layout
      %broadcasted = linalg.broadcast ins(%reduced : tensor<f32>) outs(%cst_0 : tensor<8xf32>) dimensions = [0]
      secret.yield %broadcasted : tensor<8xf32>
    } -> !secret.secret<tensor<8xf32>>
    return %0 : !secret.secret<tensor<8xf32>>
  }
}
