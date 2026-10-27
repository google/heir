// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=8 --split-input-file | FileCheck %s

// Zero-padded 1-D convolution with a cleartext filter: pad elements have no
// slots and read as zero; the all-padding group vanishes and the packed filter
// rows mask the rest.

#padded = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i - 1 and 1 <= i <= 6 and 0 <= slot <= 7 }">
#out = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i and 0 <= i <= 5 and 0 <= slot <= 7 }">
#window = affine_map<(d0, d1) -> (d0 + d1)>
#filter = affine_map<(d0, d1) -> (d1)>
#output = affine_map<(d0, d1) -> (d0)>

// CHECK: func.func @padded_conv
// CHECK-NOT: linalg.generic
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
// Filter taps packed per group.
// CHECK-DAG: arith.constant dense<{{\[\[}}2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<1x8xf32>
// CHECK-DAG: arith.constant dense<{{\[\[}}3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<1x8xf32>
// CHECK-DAG: arith.constant dense<{{\[\[}}0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<1x8xf32>
// CHECK: arith.mulf
// CHECK: tensor_ext.rotate %{{.*}}, %[[c1]]
// CHECK: arith.mulf
// CHECK: tensor_ext.rotate %{{.*}}, %[[c7]]
// CHECK: arith.mulf
// CHECK-NOT: arith.mulf
// CHECK: return
func.func @padded_conv(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #padded}) -> (!secret.secret<tensor<6xf32>> {tensor_ext.layout = #out}) {
  %taps = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %zero = arith.constant dense<0.0> : tensor<6xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #padded}) {
  ^body(%input: tensor<8xf32>):
    %init = tensor_ext.assign_layout %zero <layout = #out> {tensor_ext.layout = #out} : tensor<6xf32>
    %1 = linalg.generic {indexing_maps = [#window, #filter, #output], iterator_types = ["parallel", "reduction"], tensor_ext.layout = #out} ins(%input, %taps : tensor<8xf32>, tensor<3xf32>) outs(%init : tensor<6xf32>) {
    ^bb0(%x: f32, %w: f32, %acc: f32):
      %2 = arith.mulf %x, %w : f32
      %3 = arith.addf %acc, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<6xf32>
    secret.yield %1 : tensor<6xf32>
  } -> (!secret.secret<tensor<6xf32>> {tensor_ext.layout = #out})
  return %0 : !secret.secret<tensor<6xf32>>
}

// -----

// A concatenation as an index-dependent select: linalg.index becomes a
// cleartext iota input.

#cat = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 7 and 0 <= slot <= 7 }">
#map = affine_map<(d0) -> (d0)>

// CHECK: func.func @concat_as_index_select
// CHECK-NOT: linalg
// CHECK: %[[cond:.*]] = arith.constant dense<{{\[\[}}true, true, true, true, false, false, false, false]]> : tensor<1x8xi1>
// CHECK: arith.select %[[cond]], %{{.*}}, %{{.*}} : tensor<1x8xi1>, tensor<1x8xf32>
func.func @concat_as_index_select(%a: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #cat}, %b: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #cat}) -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #cat}) {
  %empty = tensor.empty() : tensor<8xf32>
  %c4 = arith.constant 4 : index
  %0 = secret.generic(%a: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #cat}, %b: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #cat}) {
  ^body(%pa: tensor<8xf32>, %pb: tensor<8xf32>):
    %1 = tensor_ext.assign_layout %empty <layout = #cat> {tensor_ext.layout = #cat} : tensor<8xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], tensor_ext.layout = #cat} ins(%pa, %pb : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) {
    ^bb0(%va: f32, %vb: f32, %out: f32):
      %i = linalg.index 0 : index
      %lt = arith.cmpi slt, %i, %c4 : index
      %v = arith.select %lt, %va, %vb : f32
      linalg.yield %v : f32
    } -> tensor<8xf32>
    secret.yield %2 : tensor<8xf32>
  } -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #cat})
  return %0 : !secret.secret<tensor<8xf32>>
}

// -----

// Division by a cleartext: x is reversed against the output, so four groups
// each write two slots and gather the divisor there, zero elsewhere. The kernel
// multiplies by the gathered reciprocal, zero where the divisor is zero.

#reversed = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = 7 - i and 0 <= i <= 7 and 0 <= slot <= 7 }">
#rowmajor = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i and 0 <= i <= 7 and 0 <= slot <= 7 }">
#id = affine_map<(d0) -> (d0)>

// CHECK: func.func @divide_by_cleartext
// CHECK-NOT: arith.divf
// CHECK: arith.mulf
// CHECK-NOT: arith.divf
// CHECK: return
func.func @divide_by_cleartext(%x: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #reversed}) -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #rowmajor}) {
  %c = arith.constant dense<[1.0, 2.0, 4.0, 8.0, 0.0, 2.0, 4.0, 8.0]> : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #reversed}) {
  ^body(%in: tensor<8xf32>):
    %1 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"], tensor_ext.layout = #rowmajor} ins(%in, %c : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %2 = arith.divf %a, %b : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
    secret.yield %1 : tensor<8xf32>
  } -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #rowmajor})
  return %0 : !secret.secret<tensor<8xf32>>
}

// -----

// A dead op in the body that the kernel does not support is dropped.

#rm = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i and 0 <= i <= 7 and 0 <= slot <= 7 }">
#id1 = affine_map<(d0) -> (d0)>

// CHECK: func.func @dead_extract
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @dead_extract(%x: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #rm}) -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #rm}) {
  %t = arith.constant dense<1.0> : tensor<4xf32>
  %e = tensor.empty() : tensor<8xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #rm}) {
  ^body(%in: tensor<8xf32>):
    %1 = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"], tensor_ext.layout = #rm} ins(%in : tensor<8xf32>) outs(%e : tensor<8xf32>) {
    ^bb0(%a: f32, %o: f32):
      %c0 = arith.constant 0 : index
      %unused = tensor.extract %t[%c0] : tensor<4xf32>
      %2 = arith.addf %a, %a : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
    secret.yield %1 : tensor<8xf32>
  } -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #rm})
  return %0 : !secret.secret<tensor<8xf32>>
}
