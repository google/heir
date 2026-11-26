#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gelu(%arg0: tensor<1x64xf32> {secret.secret}) -> tensor<1x64xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 1.41421354 : f32
    %0 = tensor.empty() : tensor<1x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.divf %in, %cst_1 : f32
      %3 = math.erf %2 : f32
      %4 = arith.addf %3, %cst : f32
      %5 = arith.mulf %4, %cst_0 : f32
      %6 = arith.mulf %in, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<1x64xf32>
    return %1 : tensor<1x64xf32>
  }
}
