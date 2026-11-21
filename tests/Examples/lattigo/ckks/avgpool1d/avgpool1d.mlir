#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @avgpool1d(%arg0: tensor<1x4x16xf32> {secret.secret}) -> tensor<1x4x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [1, 4, 1, 16] : tensor<1x4x16xf32> into tensor<1x4x1x16xf32>
    %0 = tensor.empty() : tensor<1x4x1x8xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    %2 = tensor.empty() : tensor<1x2xf32>
    %3 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<[1, 2]> : vector<2xi64>} ins(%expanded, %2 : tensor<1x4x1x16xf32>, tensor<1x2xf32>) outs(%1 : tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x4x1x8xf32>) outs(%0 : tensor<1x4x1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.divf %in, %cst_0 : f32
      linalg.yield %5 : f32
    } -> tensor<1x4x1x8xf32>
    %collapsed = tensor.collapse_shape %4 [[0], [1, 2], [3]] : tensor<1x4x1x8xf32> into tensor<1x4x8xf32>
    return %collapsed : tensor<1x4x8xf32>
  }
}
