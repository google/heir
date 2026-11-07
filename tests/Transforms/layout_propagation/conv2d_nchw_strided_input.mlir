// RUN: heir-opt --layout-propagation --verify-diagnostics --split-input-file %s

// A conv reading a strided conv's pixel-shuffled result is rejected.

func.func @conv2d_stride_chain(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x1x3x3xf32>> {
  %cst1 = arith.constant dense<0.0> : tensor<1x4x5x5xf32>
  %filter1 = arith.constant dense<1.0> : tensor<4x1x2x2xf32>
  %cst2 = arith.constant dense<0.0> : tensor<1x1x3x3xf32>
  %filter2 = arith.constant dense<1.0> : tensor<1x4x3x3xf32>

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter1 : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>)
      outs(%cst1 : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    // expected-error@+1 {{Reading the result of a strided convolution or average pool is not supported}}
    %2 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
      ins(%1, %filter2 : tensor<1x4x5x5xf32>, tensor<1x4x3x3xf32>)
      outs(%cst2 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    secret.yield %2 : tensor<1x1x3x3xf32>
  } -> !secret.secret<tensor<1x1x3x3xf32>>
  return %0 : !secret.secret<tensor<1x1x3x3xf32>>
}

// -----

func.func @conv2d_stride_elementwise_stride(%arg0: !secret.secret<tensor<1x1x20x20xf32>>) -> !secret.secret<tensor<1x6x5x5xf32>> {
  %cst1 = arith.constant dense<0.0> : tensor<1x4x10x10xf32>
  %filter1 = arith.constant dense<1.0> : tensor<4x1x2x2xf32>
  %cst2 = arith.constant dense<0.0> : tensor<1x6x5x5xf32>
  %filter2 = arith.constant dense<1.0> : tensor<6x4x2x2xf32>

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x20x20xf32>>) {
  ^body(%input0: tensor<1x1x20x20xf32>):
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter1 : tensor<1x1x20x20xf32>, tensor<4x1x2x2xf32>)
      outs(%cst1 : tensor<1x4x10x10xf32>) -> tensor<1x4x10x10xf32>
    %2 = arith.mulf %1, %1 : tensor<1x4x10x10xf32>
    // expected-error@+1 {{Reading the result of a strided convolution or average pool is not supported}}
    %3 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%2, %filter2 : tensor<1x4x10x10xf32>, tensor<6x4x2x2xf32>)
      outs(%cst2 : tensor<1x6x5x5xf32>) -> tensor<1x6x5x5xf32>
    secret.yield %3 : tensor<1x6x5x5xf32>
  } -> !secret.secret<tensor<1x6x5x5xf32>>
  return %0 : !secret.secret<tensor<1x6x5x5xf32>>
}
