// RUN: heir-opt --new-layout-propagation --fold-convert-layout-into-assign-layout %s | FileCheck %s

// CHECK: #kernel = #secret.kernel<name = "Conv2dMatvec", force = false>
// CHECK: @conv2d
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<5x5xf32>> {tensor_ext.layout = [[rm_layout:.*]]}) ->
func.func @conv2d(%arg0: !secret.secret<tensor<5x5xf32>>) -> !secret.secret<tensor<3x3xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
  // CHECK: %[[out:.*]] = arith.constant dense<0.00
  // CHECK: %[[filter:.*]] = arith.constant
  // CHECK-SAME: tensor<3x3xf32>

  // Assign a layout to the matrix and bias
  // CHECK-DAG: tensor_ext.assign_layout %[[filter]]
  // CHECK-DAG: tensor_ext.assign_layout %[[out]]
  %cst_0 = arith.constant dense<2.0> : tensor<3x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<5x5xf32>>) {
  ^body(%input0: tensor<5x5xf32>):
    // CHECK: linalg.conv_2d
    // CHECK-SAME: secret.kernel = #kernel
    %1 = linalg.conv_2d ins(%input0, %cst_0 : tensor<5x5xf32>, tensor<3x3xf32>) outs(%cst : tensor<3x3xf32>) -> tensor<3x3xf32>
    secret.yield %1 : tensor<3x3xf32>
    // CHECK: secret.yield
    // CHECK-NEXT: -> (!secret.secret<tensor<3x3xf32>> {tensor_ext.layout = [[rm_layout2:.*]]})
  } -> !secret.secret<tensor<3x3xf32>>
  return %0 : !secret.secret<tensor<3x3xf32>>
}
