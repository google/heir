#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func @mean(%arg0: tensor<1x64xf32> {secret.secret}) -> tensor<1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 6.400000e+01 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x64xf32>) outs(%1 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<1xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%2 : tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.divf %in, %cst_0 : f32
      linalg.yield %4 : f32
    } -> tensor<1xf32>
    return %3 : tensor<1xf32>
  }
}
