module {
  func.func @maxpool1d(%arg0: tensor<1x3x16xf32> {secret.secret}) -> tensor<1x3x8xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [1, 3, 1, 16] : tensor<1x3x16xf32> into tensor<1x3x1x16xf32>
    %0 = tensor.empty() : tensor<1x3x1x8xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x3x1x8xf32>) -> tensor<1x3x1x8xf32>
    %2 = tensor.empty() : tensor<1x2xf32>
    %3 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<[1, 2]> : vector<2xi64>} ins(%expanded, %2 : tensor<1x3x1x16xf32>, tensor<1x2xf32>) outs(%1 : tensor<1x3x1x8xf32>) -> tensor<1x3x1x8xf32>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2], [3]] : tensor<1x3x1x8xf32> into tensor<1x3x8xf32>
    return %collapsed : tensor<1x3x8xf32>
  }
}
