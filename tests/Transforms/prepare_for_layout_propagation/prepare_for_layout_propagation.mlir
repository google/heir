// RUN: heir-opt --prepare-for-layout-propagation %s | FileCheck %s

#map_proj = affine_map<(d0, d1) -> (d0)>
#map_id = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: @raise_broadcast
func.func @raise_broadcast(%x: tensor<4xf32>) -> tensor<4x8xf32> {
  %init = tensor.empty() : tensor<4x8xf32>
  // CHECK: linalg.broadcast
  // CHECK-SAME: dimensions = [1]
  // CHECK-NOT: linalg.generic
  %0 = linalg.generic
      {indexing_maps = [#map_proj, #map_id],
       iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<4xf32>) outs(%init : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK: @fold_divf_by_splat
func.func @fold_divf_by_splat(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: arith.constant dense<2.500000e-01>
  // CHECK: arith.mulf
  // CHECK-NOT: arith.divf
  %cst = arith.constant dense<4.0> : tensor<4x8xf32>
  %0 = arith.divf %x, %cst : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK: @keep_nonsplat_divf
func.func @keep_nonsplat_divf(%x: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: arith.divf
  %cst = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = arith.divf %x, %cst : tensor<2xf32>
  return %0 : tensor<2xf32>
}
