// RUN: heir-opt --layout-optimization --canonicalize %s -split-input-file | FileCheck %s

!s_ty = !secret.secret<tensor<32xi16>>

#map = #tensor_ext.layout<map = (d0) -> (d0)>
#map1 = #tensor_ext.layout<map = (d0) -> ((d0 + 1) mod 32)>
#map2 = #tensor_ext.layout<map = (d0) -> ((d0 + 2) mod 32)>

module {
  // CHECK: func @no_hoist
  // CHECK: arith.addi
  // CHECK-COUNT-1: tensor_ext.convert_layout
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: arith.addi
  // CHECK: return
  func.func @no_hoist(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>, %input2: tensor<32xi16>):
      %y = arith.addi %input0, %input2 {tensor_ext.layout = [#map]} : tensor<32xi16>
      %yy = tensor_ext.convert_layout %y {from_layout = #map, tensor_ext.layout = [#map1], to_layout = #map1} : tensor<32xi16>
      %4 = arith.addi %yy, %input1 {tensor_ext.layout = [#map1]} : tensor<32xi16>
      secret.yield %4 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1})
    return %0 : !secret.secret<tensor<32xi16>>
  }
}
