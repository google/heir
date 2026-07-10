// RUN: heir-opt --layout-propagation %s | FileCheck %s

// Check annotations of heir.kernel_info for gap_factor propagation.

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @conv2d_chain
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-SAME: gap_factor = 1 : i64, input_shape = array<i64: 1, 1, 4, 4>, result_shape = array<i64: 1, 1, 3, 3>
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-SAME: gap_factor = 1 : i64, input_shape = array<i64: 1, 1, 3, 3>, result_shape = array<i64: 1, 1, 2, 3>
  // CHECK: return
  func.func @conv2d_chain(%arg0: !secret.secret<tensor<1x1x4x4xf32>>) -> !secret.secret<tensor<1x1x2x3xf32>> {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x1x2x1xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1x1x2x2xf32>
    %0 = tensor.empty() : tensor<1x1x3x3xf32>
    %1 = tensor.empty() : tensor<1x1x2x3xf32>
    %2 = secret.generic(%arg0: !secret.secret<tensor<1x1x4x4xf32>>) {
    ^body(%input0: tensor<1x1x4x4xf32>):
      %3 = linalg.conv_2d_nchw_fchw {strides = dense<1> : vector<2xi64>} ins(%input0, %cst_0 : tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) outs(%0 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
      %4 = linalg.conv_2d_nchw_fchw {strides = dense<1> : vector<2xi64>} ins(%3, %cst : tensor<1x1x3x3xf32>, tensor<1x1x2x1xf32>) outs(%1 : tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32>
      secret.yield %4 : tensor<1x1x2x3xf32>
    } -> !secret.secret<tensor<1x1x2x3xf32>>
    return %2 : !secret.secret<tensor<1x1x2x3xf32>>
  }
}
