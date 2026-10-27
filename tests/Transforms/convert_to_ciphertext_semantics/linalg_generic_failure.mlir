// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=min-slot-count=16 --verify-diagnostics

// Generic ops that have no layout-driven FHE kernel.

// A reduction that subtracts has no order-independent accumulation.

#mat = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 4i0 + i1 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">
#out = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = 4i0 and 0 <= i0 <= 3 and 0 <= slot <= 15 }">
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @row_difference(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #mat}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #out}) {
  %init = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #mat}) {
  ^body(%input0: tensor<4x4xf32>):
    %1 = tensor_ext.assign_layout %init <layout = #out> {tensor_ext.layout = #out} : tensor<4xf32>
    // expected-error@+2 {{a reduction must combine the out argument with a value computed from the inputs}}
    // expected-error@+1 {{failed to legalize operation 'linalg.generic'}}
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"], tensor_ext.layout = #out} ins(%input0 : tensor<4x4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.subf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    secret.yield %2 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #out})
  return %0 : !secret.secret<tensor<4xf32>>
}

// -----

// A secret scalar captured by the body is usable only when it is replicated
// in every slot; this one lives in slot 0 alone.

#vec = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 15 }">
#one = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and slot = 0 }">
#map = affine_map<(d0) -> (d0)>

func.func @single_slot_scalar(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec}, %s: !secret.secret<f32> {tensor_ext.layout = #one}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec}) {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec}, %s: !secret.secret<f32> {tensor_ext.layout = #one}) {
  ^body(%input0: tensor<4xf32>, %scalar: f32):
    %1 = tensor_ext.assign_layout %empty <layout = #vec> {tensor_ext.layout = #vec} : tensor<4xf32>
    // expected-error@+2 {{a secret scalar captured by the body must be replicated in every slot}}
    // expected-error@+1 {{failed to legalize operation 'linalg.generic'}}
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"], tensor_ext.layout = #vec} ins(%input0 : tensor<4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.mulf %in, %scalar : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    secret.yield %2 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec})
  return %0 : !secret.secret<tensor<4xf32>>
}

// -----

// A lookup at an index computed from the data (an embedding) has no affine
// form.

#vec = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 15 }">
#map = affine_map<(d0) -> (d0)>

func.func @data_dependent_lookup(%arg0: !secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec}) {
  %table = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %empty = tensor.empty() : tensor<4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec}) {
  ^body(%ids: tensor<4xi32>):
    %1 = tensor_ext.assign_layout %empty <layout = #vec> {tensor_ext.layout = #vec} : tensor<4xf32>
    // expected-error@+2 {{the generic FHE kernel requires a body of single-result elementwise-mappable ops, but found tensor.extract}}
    // expected-error@+1 {{failed to legalize operation 'linalg.generic'}}
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"], tensor_ext.layout = #vec} ins(%ids : tensor<4xi32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%id: i32, %out: f32):
      %j = arith.index_cast %id : i32 to index
      %v = tensor.extract %table[%j] : tensor<4xf32>
      linalg.yield %v : f32
    } -> tensor<4xf32>
    secret.yield %2 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #vec})
  return %0 : !secret.secret<tensor<4xf32>>
}
