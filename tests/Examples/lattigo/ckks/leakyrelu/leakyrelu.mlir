#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @leakyrelu(%arg0: tensor<1x64xf32> {secret.secret}) -> tensor<1x64xf32> {
    %cst = arith.constant 1.000000e-02 : f64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<1x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0, %arg0 : tensor<f32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.cmpf ogt, %in, %in_1 : f32
      %6 = arith.select %5, %in, %in_1 : f32
      linalg.yield %6 : f32
    } -> tensor<1x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0, %arg0 : tensor<f32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.cmpf olt, %in, %in_1 : f32
      %6 = arith.select %5, %in, %in_1 : f32
      linalg.yield %6 : f32
    } -> tensor<1x64xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.truncf %cst : f64 to f32
      %6 = arith.mulf %in, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<1x64xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %3 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.addf %in, %in_1 : f32
      linalg.yield %5 : f32
    } -> tensor<1x64xf32>
    return %4 : tensor<1x64xf32>
  }
}
