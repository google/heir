// RUN: heir-opt --split-preprocessing %s | FileCheck %s

// Tests functionality when tensor_ext.assign_layout is nested inside a region.

// CHECK: func.func @hoist_one_assign__preprocessed(
// CHECK-SAME: %[[ARG0:.*]]: !secret.secret<tensor<4xi32>> {tensor_ext.layout = {{[^}]+}}},
// CHECK-SAME: %[[ARG1:.*]]: tensor<4xi32> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4xi32>, layout = {{[^}]+}}>},
// CHECK-SAME: %[[ARG2:.*]]: tensor<4xi32> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4xi32>, layout = {{[^}]+}}>})
// CHECK:      secret.generic(%[[ARG0]]
// CHECK-NEXT: ^{{.*}}(%[[INPUT0:.*]]: tensor<4xi32>):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[INPUT0]], %[[ARG1]] : tensor<4xi32>
// CHECK-NEXT:   arith.muli %[[RESULT]], %[[ARG2]] : tensor<4xi32>
// CHECK-NEXT:   secret.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @hoist_one_assign
// CHECK-SAME: (%[[ARG0:.*]]: !secret.secret<tensor<4xi32>> {tensor_ext.layout = #[[layout:.*]]}) ->
// CHECK-NEXT:   %[[C1:.*]] = arith.constant dense<1> : tensor<4xi32>
// CHECK-DAG:   %[[ASSIGN1:.*]] = tensor_ext.assign_layout %[[C1]] {layout = #[[layout]]}
// CHECK-DAG:   %[[ASSIGN2:.*]] = tensor_ext.assign_layout %[[C1]] {layout =
// CHECK-NEXT:   %[[CALL:.*]] = call @hoist_one_assign__preprocessed(%[[ARG0]], %[[ASSIGN1]], %[[ASSIGN2]])
// CHECK-NEXT:   return %[[CALL]]
// CHECK-NEXT: }

#vec_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 0 and i0 >= 0 and 3 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">
#vec_layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 0 and i0 >= 0 and 0 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">

func.func @hoist_one_assign(%arg0: !secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec_layout}) -> (!secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec_layout}) {
  %c1 = arith.constant dense<1> : tensor<4xi32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4xi32>>) {
  ^body(%input0: tensor<4xi32>):
    %assign_layout1 = tensor_ext.assign_layout %c1 {layout = #vec_layout} : tensor<4xi32>
    %assign_layout2 = tensor_ext.assign_layout %c1 {layout = #vec_layout1} : tensor<4xi32>
    %add = arith.addi %input0, %assign_layout1 : tensor<4xi32>
    %add1 = arith.muli %add, %assign_layout2 : tensor<4xi32>
    secret.yield %add1 : tensor<4xi32>
  } -> !secret.secret<tensor<4xi32>>
  return %0 : !secret.secret<tensor<4xi32>>
}
