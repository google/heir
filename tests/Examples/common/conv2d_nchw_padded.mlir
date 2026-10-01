module {
  func.func @conv2d_nchw_padded(%arg0 : tensor<1x4x4x4xf32> {secret.secret}) -> tensor<1x4x4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %filter = arith.constant dense<[[[[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [1.0, -1.0, -0.5]], [[0.5, 1.0, -1.0], [-1.0, -0.5, 0.0], [0.0, 0.5, 1.0]], [[-0.5, 0.0, 0.5], [0.5, 1.0, -1.0], [-1.0, -0.5, 0.0]], [[1.0, -1.0, -0.5], [-0.5, 0.0, 0.5], [0.5, 1.0, -1.0]]], [[[0.0, 0.5, 1.0], [1.0, -1.0, -0.5], [-0.5, 0.0, 0.5]], [[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [1.0, -1.0, -0.5]], [[0.5, 1.0, -1.0], [-1.0, -0.5, 0.0], [0.0, 0.5, 1.0]], [[-0.5, 0.0, 0.5], [0.5, 1.0, -1.0], [-1.0, -0.5, 0.0]]], [[[1.0, -1.0, -0.5], [-0.5, 0.0, 0.5], [0.5, 1.0, -1.0]], [[0.0, 0.5, 1.0], [1.0, -1.0, -0.5], [-0.5, 0.0, 0.5]], [[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [1.0, -1.0, -0.5]], [[0.5, 1.0, -1.0], [-1.0, -0.5, 0.0], [0.0, 0.5, 1.0]]], [[[-0.5, 0.0, 0.5], [0.5, 1.0, -1.0], [-1.0, -0.5, 0.0]], [[1.0, -1.0, -0.5], [-0.5, 0.0, 0.5], [0.5, 1.0, -1.0]], [[0.0, 0.5, 1.0], [1.0, -1.0, -0.5], [-0.5, 0.0, 0.5]], [[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [1.0, -1.0, -0.5]]]]> : tensor<4x4x3x3xf32>
    %bias = arith.constant dense<[-0.1, 0.0, 0.1, 0.2]> : tensor<4xf32>
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%a: index, %b: index, %c: index, %d: index):
      tensor.yield %cst : f32
    } : tensor<1x4x4x4xf32> to tensor<1x4x6x6xf32>
    %empty = tensor.empty() : tensor<1x4x4x4xf32>
    %bcast = linalg.broadcast ins(%bias : tensor<4xf32>) outs(%empty : tensor<1x4x4x4xf32>) dimensions = [0, 2, 3]
    %out = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %filter : tensor<1x4x6x6xf32>, tensor<4x4x3x3xf32>) outs(%bcast : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    return %out : tensor<1x4x4x4xf32>
  }
}
