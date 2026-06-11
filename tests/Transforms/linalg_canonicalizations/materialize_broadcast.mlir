// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// CHECK: @basic_broadcast
func.func @basic_broadcast(%arg0: tensor<3xf32>, %arg1: tensor<3x16xf32>) -> tensor<3x16xf32> {
  %cst = tensor.empty() : tensor<3x16xf32>
  // CHECK: %[[EXPANDED:.*]] = linalg.broadcast ins(%arg0 : tensor<3xf32>) outs({{.*}}) dimensions = [1]
  // CHECK: arith.addf {{.*}}, {{.*}} : tensor<3x16xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<3xf32>, tensor<3x16xf32>) outs(%cst : tensor<3x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<3x16xf32>
  return %0 : tensor<3x16xf32>
}

// -----

// CHECK: @multiple_broadcast
func.func @multiple_broadcast(%arg0: tensor<3xf32>, %arg1: tensor<16xf32>, %arg2: tensor<3x16xf32>) -> tensor<3x16xf32> {
  %cst = tensor.empty() : tensor<3x16xf32>
  // CHECK: %[[EXPANDED1:.*]] = linalg.broadcast ins(%arg0 : tensor<3xf32>) outs({{.*}}) dimensions = [1]
  // CHECK: %[[EXPANDED2:.*]] = linalg.broadcast ins(%arg1 : tensor<16xf32>) outs({{.*}}) dimensions = [0]
  // CHECK: %[[SUM:.*]] = arith.addf {{.*}}, {{.*}}
  // CHECK: arith.addf {{.*}}, {{.*}} : tensor<3x16xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1, %arg2 : tensor<3xf32>, tensor<16xf32>, tensor<3x16xf32>) outs(%cst : tensor<3x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %1 = arith.addf %in, %in_0 : f32
    %2 = arith.addf %1, %in_1 : f32
    linalg.yield %2 : f32
  } -> tensor<3x16xf32>
  return %0 : tensor<3x16xf32>
}

// -----

// CHECK: @single_input_no_broadcast
func.func @single_input_no_broadcast(%arg0: tensor<3xf32>) -> tensor<3x16xf32> {
  %cst = tensor.empty() : tensor<3x16xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1]
  // CHECK-NOT: arith.addf
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<3xf32>) outs(%cst : tensor<3x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x16xf32>
  return %0 : tensor<3x16xf32>
}

// -----

// CHECK: @reduction_no_broadcast
func.func @reduction_no_broadcast(%arg0: tensor<3x16xf32>) -> tensor<3xf32> {
  %cst = tensor.empty() : tensor<3xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<3x16xf32>) outs(%cst : tensor<3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
