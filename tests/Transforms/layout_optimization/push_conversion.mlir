// RUN: heir-opt --layout-optimization --canonicalize %s -split-input-file | FileCheck %s

!s_ty = !secret.secret<tensor<32xi16>>

// CHECK-DAG: [[layout0:#[^ ]*]] = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">

#layout0 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">
#layout1 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 1 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">
module {
  // CHECK: func @push_conversion
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[layout0]]}, %[[arg1:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[layout0]]}, %[[arg2:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = [[layout0]]})
  func.func @push_conversion(
        %arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1},
        %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0},
        %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0})
        -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0}) {
    // CHECK: secret.generic
    %0 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0}, %arg2: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>, %input2: tensor<32xi16>):
    // CHECK: ^body(%[[input0:.*]]: tensor<32xi16>, %[[input1:.*]]: tensor<32xi16>, %[[input2:.*]]: tensor<32xi16>)
    // CHECK: %[[v1:.*]] = arith.addi %[[input0]], %[[input1]]
    // CHECK-SAME: tensor_ext.layout = [[layout0]]
    // CHECK-NEXT: arith.addi %[[v1]], %[[input2]]
    // CHECK-SAME: tensor_ext.layout = [[layout0]]
    // CHECK-NEXT: secret.yield
      %1 = tensor_ext.convert_layout %input1 {from_layout = #layout0, tensor_ext.layout = [#layout1], to_layout = #layout1} : tensor<32xi16>
      %2 = arith.addi %input0, %1 {tensor_ext.layout = #layout1} : tensor<32xi16>
      %3 = tensor_ext.convert_layout %2 {from_layout = #layout1, tensor_ext.layout = [#layout0], to_layout = #layout0} : tensor<32xi16>
      %4 = arith.addi %3, %input2 {tensor_ext.layout = #layout0} : tensor<32xi16>
      secret.yield %4 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = [#layout0]})
    return %0 : !secret.secret<tensor<32xi16>>
  }
}
