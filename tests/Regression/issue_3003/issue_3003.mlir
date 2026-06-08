// dummy comment to force rebuild
module {
  func.func @bug_minrepro(%arg0 : tensor<1x1x5x5xf32> {secret.secret}) -> tensor<1x1x3x3xf32> {
    %filter = arith.constant dense<1.0> : tensor<1x1x3x3xf32>
    %out = arith.constant dense<0.0> : tensor<1x1x3x3xf32>
    %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %filter : tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) outs(%out : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    return %0 : tensor<1x1x3x3xf32>
  }
}
