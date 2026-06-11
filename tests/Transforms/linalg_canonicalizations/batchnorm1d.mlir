// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// In this test, the axes of arg0 (1x3x16) do not align with the axes of arg1
// (3x1) and so the pattern identifies that it can collapse the shape of arg1
// before re-broadcasting it to match arg0.
#map_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_2d = affine_map<(d0, d1, d2) -> (d1, 0)>

// CHECK: @test_materialize_broadcast
func.func @test_materialize_broadcast(
    %arg0: tensor<1x3x16xf32>,
    %arg1: tensor<3x1xf32>
) -> tensor<1x3x16xf32> {
  %empty = tensor.empty() : tensor<1x3x16xf32>

  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG1:.*]] {{\[\[}}0, 1{{\]\]}} : tensor<3x1xf32> into tensor<3xf32>
  // CHECK: %[[BROADCAST:.*]] = linalg.broadcast ins(%[[COLLAPSED]] : tensor<3xf32>) outs(%{{.*}} : tensor<1x3x16xf32>) dimensions = [0, 2]
  // CHECK: %[[SUB:.*]] = arith.subf %[[ARG0:.*]], %[[BROADCAST]] : tensor<1x3x16xf32>
  // CHECK: return %[[SUB]] : tensor<1x3x16xf32>
  %result = linalg.generic {
    indexing_maps = [#map_3d, #map_2d, #map_3d],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<1x3x16xf32>, tensor<3x1xf32>) outs(%empty : tensor<1x3x16xf32>) {
  ^bb0(%in: f32, %in_elt: f32, %out: f32):
    %sub = arith.subf %in, %in_elt : f32
    linalg.yield %sub : f32
  } -> tensor<1x3x16xf32>

  return %result : tensor<1x3x16xf32>
}

// -----
