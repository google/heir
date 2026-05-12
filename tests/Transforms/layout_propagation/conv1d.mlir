// RUN: heir-opt --layout-propagation --fold-convert-layout-into-assign-layout %s | FileCheck %s

// CHECK: #kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
// CHECK: @conv1d
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<5xf32>> {tensor_ext.layout = [[rm_layout:.*]]}) ->
func.func @conv1d(%arg0: !secret.secret<tensor<5xf32>>) -> !secret.secret<tensor<3xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3xf32>
  // CHECK: %[[out:.*]] = arith.constant dense<0.00
  // CHECK: %[[filter:.*]] = arith.constant
  // CHECK-SAME: tensor<3xf32>

  // Assign a layout to the filter and input
  // CHECK-DAG: tensor_ext.assign_layout %[[filter]]
  // CHECK-DAG: tensor_ext.assign_layout %[[out]]
  %cst_0 = arith.constant dense<2.0> : tensor<3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<5xf32>>) {
  ^body(%input0: tensor<5xf32>):
    // CHECK: linalg.conv_1d
    // CHECK-SAME: secret.kernel = #kernel
    %1 = linalg.conv_1d ins(%input0, %cst_0 : tensor<5xf32>, tensor<3xf32>) outs(%cst : tensor<3xf32>) -> tensor<3xf32>
    secret.yield %1 : tensor<3xf32>
    // CHECK: secret.yield
    // CHECK-NEXT: -> (!secret.secret<tensor<3xf32>> {tensor_ext.layout = [[rm_layout2:.*]]})
  } -> !secret.secret<tensor<3xf32>>
  return %0 : !secret.secret<tensor<3xf32>>
}
