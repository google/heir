// RUN: heir-opt --layout-optimization=min-slot-count=64 %s | FileCheck %s

// Hoisting the convert_layout above the addi would change the layout the muli
// reads, so the conversion stays.

#layout0 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">
#layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 1 and 31 >= i0 >= 0 and 63 >= slot >= 0 and ct = 0 }">

// CHECK: func.func @other_users(%{{.*}} {tensor_ext.layout = #[[L0:[^}]*]]})
// CHECK: %[[sum:.*]] = arith.addi {{.*}}{tensor_ext.layout = #[[L0]]}
// CHECK: tensor_ext.convert_layout %[[sum]]
// CHECK: arith.muli %[[sum]], %[[sum]] {tensor_ext.layout = #[[L0]]}
func.func @other_users(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0}) {
  %0:2 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0}) {
  ^body(%input0: tensor<32xi16>):
    %1 = arith.addi %input0, %input0 {tensor_ext.layout = #layout0} : tensor<32xi16>
    %2 = tensor_ext.convert_layout %1 <from_layout = #layout0, to_layout = #layout1> {tensor_ext.layout = #layout1} : tensor<32xi16>
    %3 = arith.muli %1, %1 {tensor_ext.layout = #layout0} : tensor<32xi16>
    secret.yield %2, %3 : tensor<32xi16>, tensor<32xi16>
  } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout0})
  return %0#0, %0#1 : !secret.secret<tensor<32xi16>>, !secret.secret<tensor<32xi16>>
}
