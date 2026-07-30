#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @sigmoid(%arg0: tensor<1x1x32x32xf32> {secret.secret}) -> tensor<1x1x32x32xf32> {
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x32x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], domain_lower = -3.0 : f64, domain_upper = 3.0 : f64} ins(%arg0 : tensor<1x1x32x32xf32>) outs(%0 : tensor<1x1x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.negf %in : f32
      %33 = math.exp %32 : f32
      %34 = arith.addf %33, %cst_0 : f32
      %35 = arith.divf %cst_0, %34 : f32
      linalg.yield %35 : f32
    } -> tensor<1x1x32x32xf32>
    return %2 : tensor<1x1x32x32xf32>
  }
}
