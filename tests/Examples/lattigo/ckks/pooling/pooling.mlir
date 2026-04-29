#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>

module {
  func.func @pooling(%arg0: tensor<1x4x28x28xf32> {secret.secret}) -> tensor<1x4x14x14xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x4x14x14xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x4x14x14xf32>) -> tensor<1x4x14x14xf32>
    %5 = arith.constant dense<1.0> : tensor<2x2xf32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %5 : tensor<1x4x28x28xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x4x14x14xf32>) -> tensor<1x4x14x14xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x4x14x14xf32>) outs(%3 : tensor<1x4x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.divf %in, %cst_1 : f32
      linalg.yield %32 : f32
    } -> tensor<1x4x14x14xf32>
    return %7 : tensor<1x4x14x14xf32>
  }
}
