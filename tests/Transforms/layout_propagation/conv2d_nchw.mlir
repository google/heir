// RUN: heir-opt --layout-propagation %s | FileCheck %s

// Input layout is a flattened row-major layout.
// CHECK-DAG: #[[layout1:.*]] = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-10i2 - i3 + slot) mod 128 = 0 and 0 <= i2 <= 9 and 0 <= i3 <= 9 and 0 <= slot <= 1023 }">
// CHECK-DAG: #kernel = #secret.kernel<name = "MatvecDiagonal", force = false>

// CHECK: @conv2d_nchw
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<1x1x10x10xf32>> {tensor_ext.layout = #[[layout1]]}
func.func @conv2d_nchw(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x4x5x5xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x4x5x5xf32>
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>

  // CHECK: %[[res:.*]] = secret.generic
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    // CHECK: linalg.conv_2d_nchw_fchw
    // CHECK-SAME: secret.kernel = #kernel
    // CHECK-SAME: strides = dense<2> : tensor<2xi64>
    // CHECK-SAME: tensor_ext.layout
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>)
      outs(%cst : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    secret.yield %1 : tensor<1x4x5x5xf32>
    // CHECK: secret.yield
  } -> !secret.secret<tensor<1x4x5x5xf32>>
  return %0 : !secret.secret<tensor<1x4x5x5xf32>>
}
