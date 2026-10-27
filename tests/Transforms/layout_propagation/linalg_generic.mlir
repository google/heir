// RUN: heir-opt --layout-propagation=min-slot-count=16 --split-input-file %s | FileCheck %s

// Layout inference for linalg.generic ops.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// A row reduction keeps each result element in the first slot of its row:
// the (lexicographically smallest) slot of the reduced input elements.
// CHECK-DAG: #[[reduced:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = 4i0 and 0 <= i0 <= 3 }">
// CHECK: func.func @row_sum
// CHECK: linalg.generic
// CHECK-SAME: tensor_ext.layout = #[[reduced]]
// CHECK: secret.yield
// CHECK: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #[[reduced]]}
func.func @row_sum(%arg0: !secret.secret<tensor<4x4xf32>>) -> !secret.secret<tensor<4xf32>> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>>) {
  ^body(%input0: tensor<4x4xf32>):
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%input0 : tensor<4x4xf32>) outs(%zero : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %out, %in : f32
      linalg.yield %2 : f32
    } -> tensor<4xf32>
    secret.yield %1 : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}

// -----

// A structured op without a dedicated kernel (here a depthwise convolution) is
// generalized, so the generic kernel implements it. Its cleartext filter needs
// no layout.
// CHECK: func.func @depthwise_conv
// CHECK-NOT: linalg.depthwise_conv_1d_nwc_wc
// CHECK-NOT: assign_layout{{.*}}tensor<3x2xf32>
// CHECK: linalg.generic
// CHECK-SAME: tensor_ext.layout
func.func @depthwise_conv(%arg0: !secret.secret<tensor<1x6x2xf32>>) -> !secret.secret<tensor<1x4x2xf32>> {
  %filter = arith.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %zero = arith.constant dense<0.0> : tensor<1x4x2xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<1x6x2xf32>>) {
  ^body(%input0: tensor<1x6x2xf32>):
    %1 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%input0, %filter : tensor<1x6x2xf32>, tensor<3x2xf32>) outs(%zero : tensor<1x4x2xf32>) -> tensor<1x4x2xf32>
    secret.yield %1 : tensor<1x4x2xf32>
  } -> !secret.secret<tensor<1x4x2xf32>>
  return %0 : !secret.secret<tensor<1x4x2xf32>>
}

// -----

// x + x^T reads %x twice, through the identity and the transpose. A copy of
// %x for the transposed read would not fold into its encryption, since the
// other read keeps %x, so both read %x in its own layout.

#id = affine_map<(d0, d1) -> (d0, d1)>
#tr = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: func.func @x_plus_transpose
// CHECK: ^body(%[[x:.*]]: tensor<4x4xf32>):
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[x]], %[[x]] :
func.func @x_plus_transpose(%arg0: !secret.secret<tensor<4x4xf32>>) -> !secret.secret<tensor<4x4xf32>> {
  %e = tensor.empty() : tensor<4x4xf32>
  %r = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>>) {
  ^body(%x: tensor<4x4xf32>):
    %1 = linalg.generic {indexing_maps = [#id, #tr, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %x : tensor<4x4xf32>, tensor<4x4xf32>) outs(%e : tensor<4x4xf32>) {
    ^bb0(%p: f32, %q: f32, %o: f32):
      %2 = arith.addf %p, %q : f32
      linalg.yield %2 : f32
    } -> tensor<4x4xf32>
    secret.yield %1 : tensor<4x4xf32>
  } -> !secret.secret<tensor<4x4xf32>>
  return %r : !secret.secret<tensor<4x4xf32>>
}

// -----

// The secret init's layout leaves elements 0 and 7 out, but the op writes
// every element, so the result does not take the init's layout.

#padded = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i - 1 and 1 <= i <= 6 and 0 <= slot <= 15 }">
#id2 = affine_map<(d0) -> (d0)>

// CHECK-DAG: #[[padded:layout[0-9]*]] = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i - 1
// CHECK: func.func @padded_init
// CHECK: tensor_ext.convert_layout %{{.*}} <from_layout = #[[padded]], to_layout = #[[total:layout[0-9]*]]>
// CHECK: linalg.generic
// CHECK-SAME: tensor_ext.layout = #[[total]]
func.func @padded_init(%init: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #padded}) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(%init: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #padded}) {
  ^body(%o: tensor<8xf32>):
    %1 = linalg.generic {indexing_maps = [#id2], iterator_types = ["parallel"]} outs(%o : tensor<8xf32>) {
    ^bb0(%out: f32):
      %c1 = arith.constant 1.0 : f32
      %2 = arith.addf %out, %c1 : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
    secret.yield %1 : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
