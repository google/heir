module {
  func.func @conv1d_ncw_padded(%arg0 : tensor<1x2x8xf32> {secret.secret}) -> tensor<1x3x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %filter = arith.constant dense<[[[-1.0, -0.5, 0.0], [0.5, 1.0, -1.0]], [[-0.5, 0.0, 0.5], [1.0, -1.0, -0.5]], [[0.0, 0.5, 1.0], [-1.0, -0.5, 0.0]]]> : tensor<3x2x3xf32>
    %bias = arith.constant dense<[0.1, -0.2, 0.3]> : tensor<3xf32>
    %padded = tensor.pad %arg0 low[0, 0, 1] high[0, 0, 1] {
    ^bb0(%a: index, %b: index, %c: index):
      tensor.yield %cst : f32
    } : tensor<1x2x8xf32> to tensor<1x2x10xf32>
    %empty = tensor.empty() : tensor<1x3x8xf32>
    %bcast = linalg.broadcast ins(%bias : tensor<3xf32>) outs(%empty : tensor<1x3x8xf32>) dimensions = [0, 2]
    %out = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%padded, %filter : tensor<1x2x10xf32>, tensor<3x2x3xf32>) outs(%bcast : tensor<1x3x8xf32>) -> tensor<1x3x8xf32>
    return %out : tensor<1x3x8xf32>
  }
}
