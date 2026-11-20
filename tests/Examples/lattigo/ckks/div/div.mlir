#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @div(%arg0: tensor<1x64xf32> {secret.secret}, %arg1: tensor<1x64xf32> {secret.secret}) -> tensor<1x64xf32> {
    %0 = tensor.empty() : tensor<1x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.divf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<1x64xf32>
    return %1 : tensor<1x64xf32>
  }
}
