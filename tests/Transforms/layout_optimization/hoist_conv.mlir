// RUN: heir-opt --layout-optimization --canonicalize %s | FileCheck %s

#data_layout = #tensor_ext.layout<"{ [d0, d1, d2, d3] -> [0, slot] : 0 <= d0 < 1 and 0 <= d1 < 1 and 0 <= d2 < 4 and 0 <= d3 < 4 and slot = d3 + 4 * d2 + 16 * d1 + 16 * d0 }">
#data_layout_2 = #tensor_ext.layout<"{ [d0, d1, d2, d3] -> [0, slot] : 0 <= d0 < 1 and 0 <= d1 < 1 and 0 <= d2 < 4 and 0 <= d3 < 4 and slot = d2 + 4 * d3 + 16 * d1 + 16 * d0 }">
// Filter layout is large but we check that it changes
#filter_layout = #tensor_ext.layout<"{ [d0, d1, d2, d3] -> [0, slot] : 0 <= d0 < 1 and 0 <= d1 < 1 and 0 <= d2 < 2 and 0 <= d3 < 2 and slot = d3 + 2 * d2 + 4 * d1 + 4 * d0 }">

// CHECK: func.func @hoist_conv
func.func @hoist_conv(%arg0: !secret.secret<tensor<1x1x4x4xf32>> {tensor_ext.layout = #data_layout}, %arg1: tensor<1x1x2x2xf32>) -> (!secret.secret<tensor<1x1x3x3xf32>> {tensor_ext.layout = #data_layout_2}) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x1x3x3xf32>
  %1 = tensor_ext.assign_layout %arg1 {layout = #filter_layout, tensor_ext.layout = #filter_layout} : tensor<1x1x2x2xf32>
  %2 = tensor_ext.assign_layout %cst {layout = #data_layout, tensor_ext.layout = #data_layout} : tensor<1x1x3x3xf32>

  // CHECK: secret.generic
  // CHECK-NOT: tensor_ext.convert_layout
  %3 = secret.generic(%arg0 : !secret.secret<tensor<1x1x4x4xf32>>) {
  ^body(%input0: tensor<1x1x4x4xf32>):
    %4 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>,
        secret.kernel = #secret.kernel<name = "MatvecDiagonal", force = false>,
        tensor_ext.layout = #data_layout }
      ins(%input0, %1 : tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>)
      outs(%2 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %5 = tensor_ext.convert_layout %4 {from_layout = #data_layout, tensor_ext.layout = #data_layout_2, to_layout = #data_layout_2} : tensor<1x1x3x3xf32>
    secret.yield %5 : tensor<1x1x3x3xf32>
  } -> (!secret.secret<tensor<1x1x3x3xf32>> {tensor_ext.layout = #data_layout_2})
  return %3 : !secret.secret<tensor<1x1x3x3xf32>>
}
