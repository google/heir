// RUN: heir-opt --split-preprocessing --split-input-file %s | FileCheck %s

// CHECK: func.func @hoist_arg__preprocessed(
// CHECK-SAME: %[[ARG0:.*]]: !secret.secret<tensor<4xi32>> {tensor_ext.layout =
// CHECK-SAME: %[[ARG1:.*]]: tensor<4xi32>
// CHECK-SAME: {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4xi32>, layout =

// CHECK-NEXT: secret.generic(%[[ARG0]]
// CHECK-NEXT: ^{{.*}}(%[[INPUT0:.*]]: tensor<4xi32>):
// CHECK-NEXT:   arith.addi %[[INPUT0]], %[[ARG1]] : tensor<4xi32>
// CHECK-NEXT:   secret.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @hoist_arg
// CHECK-SAME: (%[[ARG0:.*]]: !secret.secret<tensor<4xi32>> {tensor_ext.layout =
// CHECK-SAME:  %[[PTXT:.*]]: tensor<4xi32>
// CHECK-NEXT:   %[[ASSIGN:.*]] = tensor_ext.assign_layout %[[PTXT]] {layout =
// CHECK-NEXT:   %[[CALL:.*]] = call @hoist_arg__preprocessed(%[[ARG0]], %[[ASSIGN]])
// CHECK-NEXT:   return %[[CALL]]
// CHECK-NEXT: }

#vec_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 0 and i0 >= 0 and 0 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">

func.func @hoist_arg(%arg0: !secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec_layout}, %ptxt: tensor<4xi32>) -> (!secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec_layout}) {
  %assign = tensor_ext.assign_layout %ptxt {layout = #vec_layout} : tensor<4xi32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4xi32>>) {
  ^body(%input0: tensor<4xi32>):
    %add = arith.addi %input0, %assign : tensor<4xi32>
    secret.yield %add : tensor<4xi32>
  } -> !secret.secret<tensor<4xi32>>
  return %0 : !secret.secret<tensor<4xi32>>
}

// -----

// CHECK: func.func @hoist_arg_and_constant__preprocessed(
// CHECK-SAME: %[[ARG0:.*]]: !secret.secret<tensor<4xi32>> {tensor_ext.layout = {{[^}]+}}},
// CHECK-SAME: %[[ARG1:.*]]: tensor<4xi32> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4xi32>, layout = {{[^}]+}}>},
// CHECK-SAME: %[[ARG2:.*]]: tensor<4xi32> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4xi32>, layout = {{[^}]+}}>})

// CHECK:      secret.generic(%[[ARG0]]
// CHECK-NEXT: ^{{.*}}(%[[INPUT0:.*]]: tensor<4xi32>):
// CHECK-NEXT:   %[[ADD:.*]] = arith.addi %[[INPUT0]], %[[ARG1]] : tensor<4xi32>
// CHECK-NEXT:   %[[MUL:.*]] = arith.muli %[[ADD]], %[[ARG2]] : tensor<4xi32>
// CHECK-NEXT:   secret.yield %[[MUL]]
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @hoist_arg_and_constant
// CHECK-SAME: (%[[ARG0:.*]]: !secret.secret<tensor<4xi32>> {tensor_ext.layout =
// CHECK-SAME:  %[[PTXT:.*]]: tensor<4xi32>
// CHECK-NEXT:   %[[CST:.*]] = arith.constant
// CHECK-NEXT:   %[[ASSIGN1:.*]] = tensor_ext.assign_layout %[[PTXT]] {layout =
// CHECK-NEXT:   %[[ASSIGN2:.*]] = tensor_ext.assign_layout %[[CST]] {layout =
// CHECK-NEXT:   %[[CALL:.*]] = call @hoist_arg_and_constant__preprocessed(%[[ARG0]], %[[ASSIGN1]], %[[ASSIGN2]])
// CHECK-NEXT:   return %[[CALL]]
// CHECK-NEXT: }

#vec_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 0 and i0 >= 0 and 0 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">

func.func @hoist_arg_and_constant(%arg0: !secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec_layout}, %ptxt: tensor<4xi32>) -> (!secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec_layout}) {
  %c1 = arith.constant dense<4> : tensor<4xi32>
  %assign = tensor_ext.assign_layout %ptxt {layout = #vec_layout} : tensor<4xi32>
  %assign_c1 = tensor_ext.assign_layout %c1 {layout = #vec_layout} : tensor<4xi32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4xi32>>) {
  ^body(%input0: tensor<4xi32>):
    %add = arith.addi %input0, %assign : tensor<4xi32>
    %mul = arith.muli %add, %assign_c1 : tensor<4xi32>
    secret.yield %mul : tensor<4xi32>
  } -> !secret.secret<tensor<4xi32>>
  return %0 : !secret.secret<tensor<4xi32>>
}

// -----

// CHECK: func.func @hoist_constant_with_computation__preprocessed(
// CHECK-SAME: %[[ARG0:.*]]: !secret.secret<tensor<4xi32>> {tensor_ext.layout = {{[^}]+}}},
// CHECK-SAME: %[[ARG1:.*]]: tensor<4xi32>,
// CHECK-SAME: %[[ARG2:.*]]: tensor<4xi32> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4xi32>, layout = {{[^}]+}}>})
// CHECK:      secret.generic(%[[ARG0]]
// CHECK-NEXT: ^{{.*}}(%[[INPUT0:.*]]: tensor<4xi32>):
// CHECK-NEXT:   %[[ADD:.*]] = arith.addi %[[INPUT0]], %[[ARG2]] : tensor<4xi32>
// CHECK-NEXT:   secret.yield %[[ADD]]
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @hoist_constant_with_computation
// CHECK-SAME: (%[[ARG0:.*]]: !secret.secret<tensor<4xi32>> {tensor_ext.layout =
// CHECK-SAME:  %[[PTXT:.*]]: tensor<4xi32>
// CHECK-NEXT:   %[[CST:.*]] = arith.constant
// CHECK-NEXT:   %[[ADD:.*]] = arith.addi %[[PTXT]], %[[CST]]
// CHECK-NEXT:   %[[ASSIGN:.*]] = tensor_ext.assign_layout %[[ADD]] {layout =
// CHECK-NEXT:   %[[CALL:.*]] = call @hoist_constant_with_computation__preprocessed(%[[ARG0]], %[[PTXT]], %[[ASSIGN]])
// CHECK-NEXT:   return %[[CALL]]
// CHECK-NEXT: }

#vec_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 0 and i0 >= 0 and 0 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">

func.func @hoist_constant_with_computation(%arg0: !secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec_layout}, %ptxt: tensor<4xi32>) -> (!secret.secret<tensor<4xi32>> {tensor_ext.layout = #vec_layout}) {
  %c1 = arith.constant dense<4> : tensor<4xi32>
  %added_constant = arith.addi %ptxt, %c1 : tensor<4xi32>
  %assign = tensor_ext.assign_layout %added_constant {layout = #vec_layout} : tensor<4xi32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4xi32>>) {
  ^body(%input0: tensor<4xi32>):
    %add = arith.addi %input0, %assign : tensor<4xi32>
    secret.yield %add : tensor<4xi32>
  } -> !secret.secret<tensor<4xi32>>
  return %0 : !secret.secret<tensor<4xi32>>
}
