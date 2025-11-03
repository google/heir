module {
  func.func @convolution(%arg0: tensor<1x1x4x4xf32> {secret.secret}, %filter: tensor<2x1x3x3xf32>) -> tensor<1x2x2x2xf32> {
    %0 = tensor.empty() : tensor<1x2x2x2xf32>
    %1 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%arg0, %filter : tensor<1x1x4x4xf32>, tensor<2x1x3x3xf32>)
      outs(%0 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %1 : tensor<1x2x2x2xf32>
  }
}
