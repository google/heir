#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @conv_pool(%arg0: tensor<1x1x8x8xf32> {secret.secret}) -> tensor<1x4x3x3xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_div = arith.constant 4.000000e+00 : f32

    %filter = arith.constant dense<[
      [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
      [[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]],
      [[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]],
      [[[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]]
    ]> : tensor<4x1x3x3xf32>

    %conv_empty = tensor.empty() : tensor<1x4x6x6xf32>
    %conv_init = linalg.fill ins(%cst_0 : f32) outs(%conv_empty : tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32>

    %conv_out = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%arg0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x3x3xf32>)
      outs(%conv_init : tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32>

    %pool_empty = tensor.empty() : tensor<1x4x3x3xf32>
    %pool_init = linalg.fill ins(%cst_0 : f32) outs(%pool_empty : tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32>

    %pool_filter = arith.constant dense<1.0> : tensor<2x2xf32>
    %pool_sum = linalg.pooling_nchw_sum
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%conv_out, %pool_filter : tensor<1x4x6x6xf32>, tensor<2x2xf32>)
      outs(%pool_init : tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32>

    %pool_out = linalg.generic
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%pool_sum : tensor<1x4x3x3xf32>)
      outs(%pool_empty : tensor<1x4x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %div = arith.divf %in, %cst_div : f32
      linalg.yield %div : f32
    } -> tensor<1x4x3x3xf32>

    return %pool_out : tensor<1x4x3x3xf32>
  }
}
