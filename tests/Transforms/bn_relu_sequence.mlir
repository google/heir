#map = affine_map<(d0) -> (d0)>

module {
  func.func @main(%arg0: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_mean1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %cst_scale1 = arith.constant dense<[0.5, 0.5, 0.5, 0.5]> : tensor<4xf32>
    %cst_bias1 = arith.constant dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf32>

    %cst_mean2 = arith.constant dense<[0.5, 0.5, 0.5, 0.5]> : tensor<4xf32>
    %cst_scale2 = arith.constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf32>
    %cst_bias2 = arith.constant dense<[0.5, 0.5, 0.5, 0.5]> : tensor<4xf32>

    %init = tensor.empty() : tensor<4xf32>

    // BatchNorm 1: (x - mean) * scale + bias
    %sub1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %cst_mean1 : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %m: f32, %out: f32):
      %0 = arith.subf %in, %m : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>

    %mul1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%sub1, %cst_scale1 : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %s: f32, %out: f32):
      %0 = arith.mulf %in, %s : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>

    %bn1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%mul1, %cst_bias1 : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %0 = arith.addf %in, %b : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>

    // ReLU 1: max(0, x)
    %relu1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%bn1 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst_0 : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>

    // BatchNorm 2
    %sub2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%relu1, %cst_mean2 : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %m: f32, %out: f32):
      %0 = arith.subf %in, %m : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>

    %mul2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%sub2, %cst_scale2 : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %s: f32, %out: f32):
      %0 = arith.mulf %in, %s : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>

    %bn2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%mul2, %cst_bias2 : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %0 = arith.addf %in, %b : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>

    // ReLU 2
    %relu2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%bn2 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst_0 : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>

    return %relu2 : tensor<4xf32>
  }
}
