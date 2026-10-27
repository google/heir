// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=8 | FileCheck %s
// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=8 | FileCheck %s --check-prefix=MASK

// A matvec with a cleartext matrix and a replicated row-major vector: one
// all-ones group per diagonal (Halevi-Shoup), with the matrix packed per group.

#vec = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (slot - i0) mod 4 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 7 }">
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>

// An all-ones mask would be a splat constant of ones; none is emitted.
// MASK-NOT: dense<1.000000e+00>
// CHECK: func.func @matvec
// CHECK-NOT: linalg.generic
// The packed diagonals, one constant per group.
// CHECK-DAG: arith.constant dense<{{\[\[}}1.000000e+00, 6.000000e+00, 1.100000e+01, 1.600000e+01, 1.000000e+00, 6.000000e+00, 1.100000e+01, 1.600000e+01]]> : tensor<1x8xf32>
// CHECK-DAG: arith.constant dense<{{\[\[}}2.000000e+00, 7.000000e+00, 1.200000e+01, 1.300000e+01, 2.000000e+00, 7.000000e+00, 1.200000e+01, 1.300000e+01]]> : tensor<1x8xf32>
// CHECK-DAG: arith.constant dense<{{\[\[}}3.000000e+00, 8.000000e+00, 9.000000e+00, 1.400000e+01, 3.000000e+00, 8.000000e+00, 9.000000e+00, 1.400000e+01]]> : tensor<1x8xf32>
// CHECK-DAG: arith.constant dense<{{\[\[}}4.000000e+00, 5.000000e+00, 1.000000e+01, 1.500000e+01, 4.000000e+00, 5.000000e+00, 1.000000e+01, 1.500000e+01]]> : tensor<1x8xf32>
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
// The four products are summed as a balanced tree.
// CHECK: arith.mulf
// CHECK: tensor_ext.rotate %{{.*}}, %[[c1]]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor_ext.rotate %{{.*}}, %[[c2]]
// CHECK: arith.mulf
// CHECK: tensor_ext.rotate %{{.*}}, %[[c3]]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK-NOT: tensor_ext.rotate
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.addf
// CHECK: return
func.func @matvec(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec}) {
  %cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf32>
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec}) {
  ^body(%input0: tensor<4xf32>):
    %2 = tensor_ext.assign_layout %zero <layout = #vec> {tensor_ext.layout = #vec} : tensor<4xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"], tensor_ext.layout = #vec} ins(%cst, %input0 : tensor<4x4xf32>, tensor<4xf32>) outs(%2 : tensor<4xf32>) {
    ^bb0(%m: f32, %v: f32, %out: f32):
      %4 = arith.mulf %m, %v : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<4xf32>
    secret.yield %3 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec})
  return %0 : !secret.secret<tensor<4xf32>>
}
