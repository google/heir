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

// CHECK: @broadcast_to_singleton
func.func @broadcast_to_singleton() -> tensor<1xi64> {
  %cst = tensor.empty() : tensor<1xi64>
  // CHECK: arith.constant dense<0> : tensor<1xi64>
  // CHECK-NOT: linalg
  %c0_i64 = arith.constant 0 : i64
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%cst : tensor<1xi64>) {
  ^bb0(%out: i64):
    linalg.yield %c0_i64 : i64
  } -> tensor<1xi64>
  return %0 : tensor<1xi64>
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

// -----

// CHECK: @broadcast_type_mismatch
func.func @broadcast_type_mismatch(%arg0: tensor<3xi32>, %arg1: tensor<3x16xf32>) -> tensor<3x16xf32> {
  %cst = tensor.empty() : tensor<3x16xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<3x16xi32>
  // CHECK: %[[EXPANDED:.*]] = linalg.broadcast ins(%arg0 : tensor<3xi32>) outs(%[[EMPTY]] : tensor<3x16xi32>)
  // CHECK: %[[SITOFP:.*]] = arith.sitofp %[[EXPANDED]] : tensor<3x16xi32> to tensor<3x16xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[SITOFP]], %arg1 : tensor<3x16xf32>
  // CHECK: return %[[ADD]]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<3xi32>, tensor<3x16xf32>) outs(%cst : tensor<3x16xf32>) {
  ^bb0(%in: i32, %in_0: f32, %out: f32):
    %1 = arith.sitofp %in : i32 to f32
    %2 = arith.addf %1, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<3x16xf32>
  return %0 : tensor<3x16xf32>
}

// -----

// A squeeze-excite scale multiplied into a feature map, as torch exports it:
// the unit spatial dims of the scale are indexed with constant zeros.

// CHECK: @channel_broadcast
func.func @channel_broadcast(%arg0: tensor<1x4x1x1xf32>, %arg1: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %cst = tensor.empty() : tensor<1x4x4x4xf32>
  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %arg0 {{.*}} : tensor<1x4x1x1xf32> into tensor<1x4xf32>
  // CHECK: %[[BCAST:.*]] = linalg.broadcast ins(%[[COLLAPSED]] : tensor<1x4xf32>) outs({{.*}} : tensor<1x4x4x4xf32>) dimensions = [2, 3]
  // CHECK: arith.mulf %[[BCAST]], %arg1 : tensor<1x4x4x4xf32>
  // CHECK-NOT: linalg.generic
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<1x4x1x1xf32>, tensor<1x4x4x4xf32>) outs(%cst : tensor<1x4x4x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// -----

// The same with a constant scale folds to a constant.

// CHECK: @constant_channel_broadcast
func.func @constant_channel_broadcast(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %scale = arith.constant dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]]> : tensor<1x4x1x1xf32>
  %cst = tensor.empty() : tensor<1x4x4x4xf32>
  // CHECK-NOT: linalg.broadcast
  // CHECK: %[[SCALE:.*]] = arith.constant dense<{{.*}}> : tensor<1x4x4x4xf32>
  // CHECK: arith.mulf %arg0, %[[SCALE]] : tensor<1x4x4x4xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%scale, %arg0 : tensor<1x4x1x1xf32>, tensor<1x4x4x4xf32>) outs(%cst : tensor<1x4x4x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// -----

// linalg.broadcast cannot transpose, so a broadcast whose remaining dims are
// out of order is left alone.

// CHECK: @transposed_channel_broadcast
func.func @transposed_channel_broadcast(%arg0: tensor<4x1x1x1xf32>, %arg1: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %cst = tensor.empty() : tensor<1x4x4x4xf32>
  // CHECK-NOT: linalg.broadcast
  // CHECK: linalg.generic
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x1x1x1xf32>, tensor<1x4x4x4xf32>) outs(%cst : tensor<1x4x4x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// -----

// The second operand reads column 0 of a 4x3 tensor. Its constant result is
// not a unit dim, so it cannot be collapsed into a broadcast, and the generic
// is left alone, including the first operand's broadcast.

// CHECK: @constant_index_into_non_unit_dim
func.func @constant_index_into_non_unit_dim(%arg0: tensor<3xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %cst = tensor.empty() : tensor<4x3xf32>
  // CHECK-NOT: linalg.broadcast
  // CHECK: linalg.generic
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<3xf32>, tensor<4x3xf32>) outs(%cst : tensor<4x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// A scalar operand has no tensor to broadcast, so the generic is left alone.

// CHECK: @scalar_operand
func.func @scalar_operand(%arg0: tensor<2x3xf32>, %arg1: f32) -> tensor<2x3xf32> {
  %cst = tensor.empty() : tensor<2x3xf32>
  // CHECK-NOT: linalg.broadcast
  // CHECK: linalg.generic
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<2x3xf32>, f32) outs(%cst : tensor<2x3xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %1 = arith.subf %in, %s : f32
    linalg.yield %1 : f32
  } -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
