// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=min-slot-count=16 | FileCheck %s

// Summing the rows of a row-major 4x4 matrix into the strided output layout
// (row i at slot 4i): one group per column offset, each rotating by the offset
// and masking slots 0, 4, 8, 12. The zero init adds nothing.

#mat = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 4i0 + i1 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">
#out = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = 4i0 and 0 <= i0 <= 3 and 0 <= slot <= 15 }">
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK: func.func @row_sum
// CHECK-NOT: linalg.generic
// CHECK-DAG: %[[mask:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00{{\]\]}}> : tensor<1x16xf32>
// The four masked terms are summed as a balanced tree.
// CHECK: arith.mulf %{{.*}}, %[[mask]]
// CHECK: tensor_ext.rotate
// CHECK: arith.mulf %{{.*}}, %[[mask]]
// CHECK: arith.addf
// CHECK: tensor_ext.rotate
// CHECK: arith.mulf %{{.*}}, %[[mask]]
// CHECK: tensor_ext.rotate
// CHECK: arith.mulf %{{.*}}, %[[mask]]
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK-NOT: tensor_ext.rotate
// CHECK-NOT: arith.addf
// CHECK: return
func.func @row_sum(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #mat}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #out}) {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #mat}) {
  ^body(%input0: tensor<4x4xf32>):
    %1 = tensor_ext.assign_layout %zero <layout = #out> {tensor_ext.layout = #out} : tensor<4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"], tensor_ext.layout = #out} ins(%input0 : tensor<4x4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    secret.yield %2 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #out})
  return %0 : !secret.secret<tensor<4xf32>>
}

// -----

// A max reduction over the rows of a row-major matrix: max is idempotent, so
// each column's group is combined into the accumulator in the slots it
// writes.

// CHECK: func.func @row_max
// CHECK-COUNT-4: arith.maximumf
// CHECK-NOT: arith.maximumf
// CHECK: return

#mat = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 4i0 + i1 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">
#out = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = 4i0 and 0 <= i0 <= 3 and 0 <= slot <= 15 }">
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @row_max(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #mat}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #out}) {
  %init = arith.constant dense<0xFF800000> : tensor<4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #mat}) {
  ^body(%input0: tensor<4x4xf32>):
    %1 = tensor_ext.assign_layout %init <layout = #out> {tensor_ext.layout = #out} : tensor<4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"], tensor_ext.layout = #out} ins(%input0 : tensor<4x4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.maximumf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    secret.yield %2 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #out})
  return %0 : !secret.secret<tensor<4xf32>>
}

// -----

// Each row accumulates a cleartext scalar the body captures from the enclosing
// function, once per column: the combined term is the captured scalar itself.

#mat = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 4i0 + i1 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">
#out = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = 4i0 and 0 <= i0 <= 3 and 0 <= slot <= 15 }">
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK: func.func @row_count
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @row_count(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #mat}, %s: f32) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #out}) {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #mat}) {
  ^body(%input0: tensor<4x4xf32>):
    %1 = tensor_ext.assign_layout %zero <layout = #out> {tensor_ext.layout = #out} : tensor<4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"], tensor_ext.layout = #out} ins(%input0 : tensor<4x4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %s : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    secret.yield %2 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #out})
  return %0 : !secret.secret<tensor<4xf32>>
}
