// RUN: heir-opt --layout-optimization=ciphertext-size=64 --canonicalize %s | FileCheck %s

// Cyclic repetition layout
#layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 1 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">
#layout1 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">
#layout2 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 2 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">

// CHECK-DAG: [[layout:#[^ ]*]] = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 1 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">
// CHECK-DAG: [[layout2:#[^ ]*]] = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 2 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">
module {
  // CHECK: func @update_uses
  // 4. Fold first tensor_ext.convert_layout's into the function argument's layout.
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[layout2]]},
  // CHECK-SAME:  %[[arg1:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[layout2]]})
  func.func @update_uses(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout2}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) {
    // CHECK-NEXT: secret.generic
    %0 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout2}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>, %input2: tensor<32xi16>):
    // CHECK-NEXT: ^body(%[[input0:[^:]*]]: tensor<32xi16>, %[[input1:[^:]*]]: tensor<32xi16>, %[[input2:[^:]*]]: tensor<32xi16>)
      // 3. Hoist %2 before %1 so arith.addi is done in layout #layout2.
      // CHECK: %[[v2:[^ ]*]] = arith.addi %[[input0]], %[[input0]]
      // CHECK-SAME: tensor_ext.layout = [[layout2]]

      %1 = arith.addi %input0, %input0 {tensor_ext.layout = #layout1} : tensor<32xi16>
      %2 = tensor_ext.convert_layout %1 {from_layout = #layout1, tensor_ext.layout = [#layout], to_layout = #layout} : tensor<32xi16>
      %3 = tensor_ext.convert_layout %1 {from_layout = #layout1, tensor_ext.layout = [#layout2], to_layout = #layout2} : tensor<32xi16>

      // 2. No change needed since no tensor_ext.convert_layout follows.
      // CHECK: %[[v3:.*]] = arith.addi %[[v2]], %[[input2]]
      // CHECK-SAME: tensor_ext.layout = [[layout2]]
      %4 = arith.addi %3, %input2 {tensor_ext.layout = #layout2} : tensor<32xi16>

      // 1. Hoist %6 before %5 so arith.addi is done in layout #layout2.
      // CHECK: arith.addi %[[v2]], %[[input1]]
      // CHECK-SAME: tensor_ext.layout = [[layout2]]
      %5 = arith.addi %2, %input1 {tensor_ext.layout = #layout} : tensor<32xi16>
      %6 = tensor_ext.convert_layout %5 {from_layout = #layout, tensor_ext.layout = [#layout2], to_layout = #layout2} : tensor<32xi16>

      // CHECK: arith.addi
      // CHECK-SAME: tensor_ext.layout = [[layout2]]
      %7 = arith.addi %4, %6 {tensor_ext.layout = #layout2} : tensor<32xi16>
      secret.yield %7 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout2})
    return %0 : !secret.secret<tensor<32xi16>>
  }
}
