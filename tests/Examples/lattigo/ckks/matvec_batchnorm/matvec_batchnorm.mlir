module {
  func.func @matvec_batchnorm(%arg0: tensor<4xf32> {secret.secret}, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    %cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf32>
    %cst_0 = tensor.empty() : tensor<4xf32>

    // Matvec
    %0 = linalg.matvec ins(%cst, %arg0 : tensor<4x4xf32>, tensor<4xf32>) outs(%cst_0 : tensor<4xf32>) -> tensor<4xf32>

    // Scale
    %1 = tensor.empty() : tensor<4xf32>
    %2 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>

    // Shift
    %3 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%2, %arg2 : tensor<4xf32>, tensor<4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4xf32>

    return %3 : tensor<4xf32>
  }
}
