// RUN: heir-opt --layout-optimization --canonicalize %s -split-input-file | FileCheck %s

!s_ty = !secret.secret<tensor<32xi16>>

// CHECK: [[map1:#[^ ]*]] = #tensor_ext.layout<map = (d0) -> (d0)>

#map = #tensor_ext.layout<map = (d0) -> ((d0 + 1) mod 32)>
#map1 = #tensor_ext.layout<map = (d0) -> (d0)>
module {
  // CHECK: func @push_conversion
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[map1]]}, %[[arg1:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[map1]]}, %[[arg2:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[map1]]})
  func.func @push_conversion(
        %arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map},
        %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1},
        %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1})
        -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}) {
    // CHECK: secret.generic
    %0 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>, %input2: tensor<32xi16>):
    // CHECK: ^body(%[[input0:.*]]: tensor<32xi16>, %[[input1:.*]]: tensor<32xi16>, %[[input2:.*]]: tensor<32xi16>)
    // CHECK: %[[v1:.*]] = arith.addi %[[input0]], %[[input1]]
    // CHECK-SAME: tensor_ext.layout = [[map1]]
    // CHECK-NEXT: arith.addi %[[v1]], %[[input2]]
    // CHECK-SAME: tensor_ext.layout = [[map1]]
    // CHECK-NEXT: secret.yield
      %1 = tensor_ext.convert_layout %input1 {from_layout = #map1, tensor_ext.layout = [#map], to_layout = #map} : tensor<32xi16>
      %2 = arith.addi %input0, %1 {tensor_ext.layout = #map} : tensor<32xi16>
      %3 = tensor_ext.convert_layout %2 {from_layout = #map, tensor_ext.layout = [#map1], to_layout = #map1} : tensor<32xi16>
      %4 = arith.addi %3, %input2 {tensor_ext.layout = #map1} : tensor<32xi16>
      secret.yield %4 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = [#map1]})
    return %0 : !secret.secret<tensor<32xi16>>
  }
}
