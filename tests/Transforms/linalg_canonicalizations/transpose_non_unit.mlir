// RUN: heir-opt --linalg-canonicalizations %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

module {
  func.func @transpose_non_unit(%arg0: tensor<2x3x4xf32>, %arg1: tensor<3x2x4xf32>) -> tensor<2x3x4xf32> {
    // CHECK: linalg.transpose
    // CHECK: arith.addf {{.*}} : tensor<2x3x4xf32>
    %out = tensor.empty() : tensor<2x3x4xf32>
    %0 = linalg.generic {
      indexing_maps = [#map0, #map1, #map0],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<2x3x4xf32>, tensor<3x2x4xf32>) outs(%out : tensor<2x3x4xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out_val: f32):
      %add = arith.addf %in0, %in1 : f32
      linalg.yield %add : f32
    } -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}
