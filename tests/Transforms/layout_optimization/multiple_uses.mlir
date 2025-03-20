// RUN: heir-opt --layout-optimization --canonicalize %s -split-input-file | FileCheck %s

// CHECK: #[[map:.*]] = affine_map<(d0) -> ((d0 + 1) mod 32)>
// CHECK: #[[map1:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[map2:.*]] = affine_map<(d0) -> ((d0 + 2) mod 32)>
#map = affine_map<(d0) -> ((d0 + 1) mod 32)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ((d0 + 2) mod 32)>
module {
  func.func @update_uses(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map2}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}) {
    // CHECK: secret.generic
    %0 = secret.generic ins(%arg0, %arg1, %arg2 : !secret.secret<tensor<32xi16>>, !secret.secret<tensor<32xi16>>, !secret.secret<tensor<32xi16>>)
      attrs = {__argattrs = [{tensor_ext.layout = #map1}, {tensor_ext.layout = #map}, {tensor_ext.layout = #map2}], __resattrs = [{tensor_ext.layout = #map2}]} {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>, %input2: tensor<32xi16>):
    // CHECK: ^body(%[[input0:.*]]: tensor<32xi16>, %[[input1:.*]]: tensor<32xi16>, %[[input2:.*]]: tensor<32xi16>)

      // 3. Hoist %2 before %1 so arith.addi is done in layout #map2.
      // CHECK: %[[v1:.*]] = tensor_ext.convert_layout %[[input0]]
      // CHECK-SAME: to_layout = #[[map2]]
      // CHECK: %[[v2:.*]] = arith.addi %[[v1]], %[[v1]]
      // CHECK-SAME: tensor_ext.layout = #[[map2]]
      %1 = arith.addi %input0, %input0 {tensor_ext.layout = #map1} : tensor<32xi16>
      %2 = tensor_ext.convert_layout %1 {from_layout = #map1, tensor_ext.layout = [#map], to_layout = #map} : tensor<32xi16>
      %3 = tensor_ext.convert_layout %1 {from_layout = #map1, tensor_ext.layout = [#map2], to_layout = #map2} : tensor<32xi16>

      // 2. No change needed since no tensor_ext.convert_layout follows.
      // CHECK: %[[v3:.*]] = arith.addi %[[v2]], %[[input2]]
      // CHECK-SAME: tensor_ext.layout = #[[map2]]
      %4 = arith.addi %3, %input2 {tensor_ext.layout = #map2} : tensor<32xi16>

      // 1. Hoist %6 before %5 so arith.addi is done in layout #map2.
      // CHECK: %[[v4:.*]] = tensor_ext.convert_layout %[[input1]]
      // CHECK: arith.addi %[[v2]], %[[v4]]
      // CHECK-SAME: tensor_ext.layout = #[[map2]]
      %5 = arith.addi %2, %input1 {tensor_ext.layout = #map} : tensor<32xi16>
      %6 = tensor_ext.convert_layout %5 {from_layout = #map, tensor_ext.layout = [#map2], to_layout = #map2} : tensor<32xi16>

      // CHECK: arith.addi
      // CHECK-SAME: tensor_ext.layout = #[[map2]]
      %7 = arith.addi %4, %6 {tensor_ext.layout = #map2} : tensor<32xi16>
      secret.yield %7 : tensor<32xi16>
    } -> !secret.secret<tensor<32xi16>>
    return %0 : !secret.secret<tensor<32xi16>>
  }
}
