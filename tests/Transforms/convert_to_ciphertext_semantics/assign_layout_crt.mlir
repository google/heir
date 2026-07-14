// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s

// CHECK: func.func private @_assign_layout_{{[0-9]+}}(%[[ARG0:.*]]: tensor<11x13xi16>) -> tensor<1x1024xi16>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<1x1024xi16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[C11:.*]] = arith.constant 11 : index
// CHECK-DAG: %[[C13:.*]] = arith.constant 13 : index
// CHECK: %[[RES:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[C1024]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (tensor<1x1024xi16>) {
// CHECK:   %[[ROW:.*]] = arith.remsi %[[K]], %[[C11]] : index
// CHECK:   %[[COL:.*]] = arith.remsi %[[K]], %[[C13]] : index
// CHECK:   %[[EXT:.*]] = tensor.extract %[[ARG0]][%[[ROW]], %[[COL]]] : tensor<11x13xi16>
// CHECK:   %[[INS:.*]] = tensor.insert %[[EXT]] into %[[ACC]][%[[C0]], %[[K]]] : tensor<1x1024xi16>
// CHECK:   scf.yield %[[INS]] : tensor<1x1024xi16>
// CHECK: }
// CHECK: return %[[RES]] : tensor<1x1024xi16>

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (78i0 + 66i1 - slot) mod 143 = 0 and 0 <= i0 <= 10 and 0 <= i1 <= 12 and 0 <= slot <= 1023 }">
module {
  func.func @test_bicyclic_assign_layout(%arg0: tensor<11x13xi16>) -> (!secret.secret<tensor<11x13xi16>> {tensor_ext.layout = #bicyclic}) {
    %0 = secret.generic(%arg0: tensor<11x13xi16>) {
    ^body(%input: tensor<11x13xi16>):
      %1 = tensor_ext.assign_layout %input {layout = #bicyclic, tensor_ext.layout = #bicyclic} : tensor<11x13xi16>
      secret.yield %1 : tensor<11x13xi16>
    } -> (!secret.secret<tensor<11x13xi16>> {tensor_ext.layout = #bicyclic})
    return %0 : !secret.secret<tensor<11x13xi16>>
  }
}

// -----

// CHECK: func.func private @_assign_layout_{{[0-9]+}}(%[[ARG0:.*]]: tensor<2x11x13xi16>) -> tensor<1x1024xi16>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<1x1024xi16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C11:.*]] = arith.constant 11 : index
// CHECK-DAG: %[[C13:.*]] = arith.constant 13 : index
// CHECK: %[[RES:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[C1024]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (tensor<1x1024xi16>) {
// CHECK:   %[[D0:.*]] = arith.remsi %[[K]], %[[C2]] : index
// CHECK:   %[[D1:.*]] = arith.remsi %[[K]], %[[C11]] : index
// CHECK:   %[[D2:.*]] = arith.remsi %[[K]], %[[C13]] : index
// CHECK:   %[[EXT:.*]] = tensor.extract %[[ARG0]][%[[D0]], %[[D1]], %[[D2]]] : tensor<2x11x13xi16>
// CHECK:   %[[INS:.*]] = tensor.insert %[[EXT]] into %[[ACC]][%[[C0]], %[[K]]] : tensor<1x1024xi16>
// CHECK:   scf.yield %[[INS]] : tensor<1x1024xi16>
// CHECK: }
// CHECK: return %[[RES]] : tensor<1x1024xi16>

#tricyclic = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (143i0 + 78i1 + 66i2 - slot) mod 286 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 10 and 0 <= i2 <= 12 and 0 <= slot <= 1023 }">
module {
  func.func @test_tricyclic_assign_layout(%arg0: tensor<2x11x13xi16>) -> (!secret.secret<tensor<2x11x13xi16>> {tensor_ext.layout = #tricyclic}) {
    %0 = secret.generic(%arg0: tensor<2x11x13xi16>) {
    ^body(%input: tensor<2x11x13xi16>):
      %1 = tensor_ext.assign_layout %input {layout = #tricyclic, tensor_ext.layout = #tricyclic} : tensor<2x11x13xi16>
      secret.yield %1 : tensor<2x11x13xi16>
    } -> (!secret.secret<tensor<2x11x13xi16>> {tensor_ext.layout = #tricyclic})
    return %0 : !secret.secret<tensor<2x11x13xi16>>
  }
}
