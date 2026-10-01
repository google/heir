// RUN: heir-opt --layout-propagation %s | FileCheck %s

// Regression test for a layout bug around the gap-structured (pixel
// shuffled) result layout of strided convolutions: the conv kernel adds its
// init (bias) operand directly to the kernel output, which is packed per the
// gap layout — the init must be re-packed into that layout instead of
// keeping its default row-major layout.

// The composed gap result layout of the stride-2 conv (existential chain
// from the pixel-shuffle), assigned to the init below.
// CHECK-DAG: #[[init_layout:layout[0-9]*]] = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : exists (e1, e2, e3, e4:

// CHECK: @conv2d_gap_init
func.func @conv2d_gap_init(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x4x5x5xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x5x5xf32>

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    // The bias (init) is re-packed into the conv kernel's gap result layout
    // rather than keeping its default row-major layout.
    // CHECK: %[[init:[^ ]+]] = tensor_ext.assign_layout
    // CHECK-SAME: layout = #[[init_layout]]
    // CHECK: linalg.conv_2d_nchw_fchw
    // CHECK-SAME: outs(%[[init]]
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>)
      outs(%bias : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    secret.yield %1 : tensor<1x4x5x5xf32>
  } -> !secret.secret<tensor<1x4x5x5xf32>>
  return %0 : !secret.secret<tensor<1x4x5x5xf32>>
}
