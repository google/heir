// RUN: heir-opt --layout-optimization=ciphertext-size=64 --canonicalize %s | FileCheck %s

// CHECK-DAG: [[map:#[^ ]*]] = #tensor_ext.layout<map = (d0) -> ((d0 + 1) mod 32)>
// CHECK-DAG: [[map2:#[^ ]*]] = #tensor_ext.layout<map = (d0) -> ((d0 + 2) mod 32)>
#map =  #tensor_ext.layout<map = (d0) -> ((d0 + 1) mod 32)>
#map1 = #tensor_ext.layout<map = (d0) -> (d0)>
#map2 = #tensor_ext.layout<map = (d0) -> ((d0 + 2) mod 32)>
module {
  // CHECK: func @update_uses
  // 4. Fold first tensor_extr.convert_layout's into the function argument's layout.
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[map2]]},
  // CHECK-SAME:  %[[arg1:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[map2]]})
  func.func @update_uses(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map2}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}) {
    // CHECK-NEXT: secret.generic
    %0 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map2}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>, %input2: tensor<32xi16>):
    // CHECK-NEXT: ^body(%[[input0:[^:]*]]: tensor<32xi16>, %[[input1:[^:]*]]: tensor<32xi16>, %[[input2:[^:]*]]: tensor<32xi16>)
      // 3. Hoist %2 before %1 so arith.addi is done in layout #map2.
      // CHECK: %[[v2:[^ ]*]] = arith.addi %[[input0]], %[[input0]]
      // CHECK-SAME: tensor_ext.layout = [[map2]]

      %1 = arith.addi %input0, %input0 {tensor_ext.layout = #map1} : tensor<32xi16>
      %2 = tensor_ext.convert_layout %1 {from_layout = #map1, tensor_ext.layout = [#map], to_layout = #map} : tensor<32xi16>
      %3 = tensor_ext.convert_layout %1 {from_layout = #map1, tensor_ext.layout = [#map2], to_layout = #map2} : tensor<32xi16>

      // 2. No change needed since no tensor_ext.convert_layout follows.
      // CHECK: %[[v3:.*]] = arith.addi %[[v2]], %[[input2]]
      // CHECK-SAME: tensor_ext.layout = [[map2]]
      %4 = arith.addi %3, %input2 {tensor_ext.layout = #map2} : tensor<32xi16>

      // 1. Hoist %6 before %5 so arith.addi is done in layout #map2.
      // CHECK: arith.addi %[[v2]], %[[input1]]
      // CHECK-SAME: tensor_ext.layout = [[map2]]
      %5 = arith.addi %2, %input1 {tensor_ext.layout = #map} : tensor<32xi16>
      %6 = tensor_ext.convert_layout %5 {from_layout = #map, tensor_ext.layout = [#map2], to_layout = #map2} : tensor<32xi16>

      // CHECK: arith.addi
      // CHECK-SAME: tensor_ext.layout = [[map2]]
      %7 = arith.addi %4, %6 {tensor_ext.layout = #map2} : tensor<32xi16>
      secret.yield %7 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map2})
    return %0 : !secret.secret<tensor<32xi16>>
  }
}
