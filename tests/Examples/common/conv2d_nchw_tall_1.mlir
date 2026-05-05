module {
  func.func @conv2d_nchw(%arg0 : tensor<1x1x4x4xf32> {secret.secret}) -> tensor<1x8x2x2xf32> {
    %filter = arith.constant dense<1.0> : tensor<8x1x2x2xf32>
    %out = arith.constant dense<0.0> : tensor<1x8x2x2xf32>
    %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %filter : tensor<1x1x4x4xf32>, tensor<8x1x2x2xf32>) outs(%out : tensor<1x8x2x2xf32>) -> tensor<1x8x2x2xf32>
    return %0 : tensor<1x8x2x2xf32>
  }
}
