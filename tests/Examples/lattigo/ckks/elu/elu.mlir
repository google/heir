#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elu(%arg0: tensor<1x64xf32> {secret.secret}) -> tensor<1x64xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst, %arg0 : tensor<f32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %6 = arith.cmpf ogt, %in, %in_1 : f32
      %7 = arith.select %6, %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<1x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst, %arg0 : tensor<f32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %6 = arith.cmpf olt, %in, %in_1 : f32
      %7 = arith.select %6, %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<1x64xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.exp %in : f32
      linalg.yield %6 : f32
    } -> tensor<1x64xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.subf %in, %cst_0 : f32
      linalg.yield %6 : f32
    } -> tensor<1x64xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %4 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %6 = arith.addf %in, %in_1 : f32
      linalg.yield %6 : f32
    } -> tensor<1x64xf32>
    return %5 : tensor<1x64xf32>
  }
}
