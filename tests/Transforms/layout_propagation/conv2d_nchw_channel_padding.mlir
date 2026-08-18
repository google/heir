// RUN: heir-opt --layout-propagation %s | FileCheck %s

// A strided conv whose output channel count is not a multiple of gap^2. The
// pixel-shuffled layout folds gap^2 = 4 channels into each 2x2 spatial block,
// so the layout reserves ceil(6 / 4) = 2 blocks and leaves 2 channels of the
// last block empty.

// CHECK: @conv2d_nchw_six_channels
func.func @conv2d_nchw_six_channels(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x6x5x5xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x6x5x5xf32>
  %filter = arith.constant dense<2.500000e-01> : tensor<6x1x2x2xf32>

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    // CHECK: linalg.conv_2d_nchw_fchw
    // The result reserves ceil(6 / 4) = 2 channel blocks of 10x10 sub-pixels.
    // CHECK-SAME: heir.kernel_info = {gap_factor = 2 : i64, input_shape = array<i64: 1, 1, 10, 10>, result_shape = array<i64: 1, 2, 10, 10>}
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x1x10x10xf32>, tensor<6x1x2x2xf32>)
      outs(%cst : tensor<1x6x5x5xf32>) -> tensor<1x6x5x5xf32>
    secret.yield %1 : tensor<1x6x5x5xf32>
  } -> !secret.secret<tensor<1x6x5x5xf32>>
  return %0 : !secret.secret<tensor<1x6x5x5xf32>>
}

// -----

// The 1-D analogue: the shuffle folds `stride` channels into each gap, so 3
// output channels at stride 2 reserve 4.

// CHECK: @conv1d_ncw_three_channels
func.func @conv1d_ncw_three_channels(%arg0: !secret.secret<tensor<1x1x16xf32>>) -> !secret.secret<tensor<1x3x8xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x3x8xf32>
  %filter = arith.constant dense<2.500000e-01> : tensor<3x1x2xf32>

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x16xf32>>) {
  ^body(%input0: tensor<1x1x16xf32>):
    // CHECK: linalg.conv_1d_ncw_fcw
    // CHECK-SAME: heir.kernel_info = {gap_factor = 2 : i64, input_shape = array<i64: 1, 1, 16>, result_shape = array<i64: 1, 2, 16>}
    %1 = linalg.conv_1d_ncw_fcw
      { dilations = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64> }
      ins(%input0, %filter : tensor<1x1x16xf32>, tensor<3x1x2xf32>)
      outs(%cst : tensor<1x3x8xf32>) -> tensor<1x3x8xf32>
    secret.yield %1 : tensor<1x3x8xf32>
  } -> !secret.secret<tensor<1x3x8xf32>>
  return %0 : !secret.secret<tensor<1x3x8xf32>>
}
