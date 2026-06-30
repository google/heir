module {
  func.func @conv2d_chain(%arg0: tensor<1x1x4x4xf32> {secret.secret}) -> tensor<1x1x2x3xf32> {
    %cst1 = arith.constant dense<1.0> : tensor<1x1x2x2xf32>
    %init1 = tensor.empty() : tensor<1x1x3x3xf32>
    // conv 2x2 on 4x4 data -> 1x1x3x3
    // output shape is 3x3 but fits in the 4x4 output row shape
    %conv1 = linalg.conv_2d_nchw_fchw {strides = dense<1> : vector<2xi64>}
      ins(%arg0, %cst1 : tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>)
      outs(%init1 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // conv 2x1 on 3x3 data -> produces 2x3
    // the next convolution expects a stride of size 3 for the rows
    // but the actual stride is 4
    %cst2 = arith.constant dense<1.0> : tensor<1x1x2x1xf32>
    %init2 = tensor.empty() : tensor<1x1x2x3xf32>
    %conv2 = linalg.conv_2d_nchw_fchw {strides = dense<1> : vector<2xi64>}
      ins(%conv1, %cst2 : tensor<1x1x3x3xf32>, tensor<1x1x2x1xf32>)
      outs(%init2 : tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32>

    return %conv2 : tensor<1x1x2x3xf32>
  }
}
