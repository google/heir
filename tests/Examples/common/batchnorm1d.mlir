#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, 0)>
module {
  func.func @batchnorm1d(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<i64>, %arg3: tensor<1x3x16xf32> {secret.secret}) -> tensor<1x3x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_torch.float32_1> : tensor<3xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_3 = arith.constant 1.000000e-05 : f64
    %0 = tensor.empty() : tensor<3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1 : tensor<3xf32>) outs(%0 : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.truncf %cst_3 : f64 to f32
      %10 = arith.addf %in, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<3xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<3xf32>) outs(%0 : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = math.sqrt %in : f32
      linalg.yield %9 : f32
    } -> tensor<3xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%2 : tensor<3xf32>) outs(%0 : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.cmpf one, %in, %cst : f32
      cf.assert %9, "unimplemented: tensor with zero element"
      %10 = arith.divf %cst_0, %in : f32
      linalg.yield %10 : f32
    } -> tensor<3xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [3, 1] : tensor<3xf32> into tensor<3x1xf32>
    %expanded_4 = tensor.expand_shape %3 [[0, 1]] output_shape [3, 1] : tensor<3xf32> into tensor<3x1xf32>
    %4 = tensor.empty() : tensor<1x3x16xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3, %expanded : tensor<1x3x16xf32>, tensor<3x1xf32>) outs(%4 : tensor<1x3x16xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %9 = arith.subf %in, %in_7 : f32
      linalg.yield %9 : f32
    } -> tensor<1x3x16xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %expanded_4 : tensor<1x3x16xf32>, tensor<3x1xf32>) outs(%4 : tensor<1x3x16xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %9 = arith.mulf %in, %in_7 : f32
      linalg.yield %9 : f32
    } -> tensor<1x3x16xf32>
    %expanded_5 = tensor.expand_shape %cst_2 [[0, 1]] output_shape [3, 1] : tensor<3xf32> into tensor<3x1xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %expanded_5 : tensor<1x3x16xf32>, tensor<3x1xf32>) outs(%4 : tensor<1x3x16xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %9 = arith.mulf %in, %in_7 : f32
      linalg.yield %9 : f32
    } -> tensor<1x3x16xf32>
    %expanded_6 = tensor.expand_shape %cst_1 [[0, 1]] output_shape [3, 1] : tensor<3xf32> into tensor<3x1xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %expanded_6 : tensor<1x3x16xf32>, tensor<3x1xf32>) outs(%4 : tensor<1x3x16xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %9 = arith.addf %in, %in_7 : f32
      linalg.yield %9 : f32
    } -> tensor<1x3x16xf32>
    return %8 : tensor<1x3x16xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_torch.float32_1: "0x04000000000000000000000000000000",
      torch_tensor_3_torch.float32: "0x040000000000803F0000803F0000803F"
    }
  }
#-}
