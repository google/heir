module {
  func.func @pooling(%arg0: tensor<1x2x6x6xf32> {secret.secret}) -> tensor<1x2x3x3xf32> {
    %5 = tensor.empty() : tensor<2x2xf32>
    %11 = tensor.empty() : tensor<1x2x3x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
    %13 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %5 : tensor<1x2x6x6xf32>, tensor<2x2xf32>) outs(%12 : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
    return %13 : tensor<1x2x3x3xf32>
  }
}
