// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=16 | FileCheck %s

// Tensor is repeated twice, so the packed cleartext should use two nonzero slots
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (slot - i0) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 15 }">
// Scalar is repeated throughout the ciphertext
#scalar_layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 15 }">

// CHECK:   func.func @insert_cleartext_into_secret(%[[arg0:[^:]*]]: !secret.secret<tensor<1x16xi16>>
// CHECK-DAG:    %[[c11:.*]] = arith.constant 11 : index
// CHECK-DAG:    %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[c0_i16:.*]] = arith.constant 0 : i16
// CHECK-DAG:    %[[c3:.*]] = arith.constant 3 : index
// CHECK-DAG:    %[[c7:.*]] = arith.constant 7 : i16
// CHECK-DAG:    %[[dense1:.*]] = arith.constant dense<1> : tensor<1x16xi16>
// CHECK-DAG:    %[[dense0:.*]] = arith.constant dense<0> : tensor<1x16xi16>
// CHECK:    %[[res:.*]] = secret.generic(%[[arg0]]: !secret.secret<tensor<1x16xi16>>) {
// CHECK:    ^body(%[[input:.*]]: tensor<1x16xi16>):
// CHECK:      %[[pt1:.*]] = tensor.insert %[[c7]] into %[[dense0]][%[[c0_index]], %[[c3]]]
// CHECK:      %[[mask1:.*]] = tensor.insert %[[c0_i16]] into %[[dense1]][%c0, %c3]
// CHECK:      %[[pt2:.*]] = tensor.insert %[[c7]] into %[[pt1]][%[[c0_index]], %[[c11]]]
// CHECK:      %[[mask2:.*]] = tensor.insert %[[c0_i16]] into %[[mask1]][%[[c0_index]], %[[c11]]]
// CHECK:      arith.muli
// CHECK:      arith.addi
// CHECK:    return %[[res]]

func.func @insert_cleartext_into_secret(%arg0: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) {
  %insertIndex = arith.constant 3 : index
  %c7 = arith.constant 7 : i16
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) {
  ^body(%input0: tensor<8xi16>):
    %0 = tensor.insert %c7 into %input0[%insertIndex] {tensor_ext.layout = #layout} : tensor<8xi16>
    secret.yield %0 : tensor<8xi16>
  } -> (!secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout})
  return %0 : !secret.secret<tensor<8xi16>>
}

// CHECK:   func.func @insert_cleartext_into_secret_dynamic(%[[arg0:[^:]*]]: !secret.secret<tensor<1x16xi16>>
// CHECK-DAG:    %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[c0_i16:.*]] = arith.constant 0 : i16
// CHECK-DAG:    %[[c7:.*]] = arith.constant 7 : i16
// CHECK-DAG:    %[[dense1:.*]] = arith.constant dense<1> : tensor<1x16xi16>
// CHECK-DAG:    %[[dense0:.*]] = arith.constant dense<0> : tensor<1x16xi16>
// CHECK:    %[[res:.*]] = secret.generic(%[[arg0]]: !secret.secret<tensor<1x16xi16>>) {
// CHECK:    ^body(%[[input:.*]]: tensor<1x16xi16>):

//           The loop that implements the codegen for the masks
// CHECK:      scf.for
// CHECK:      scf.if
// CHECK:      tensor.insert
// CHECK:      tensor.insert
// CHECK:      scf.yield
// CHECK:      } else {
// CHECK:      scf.yield
// CHECK:      }
// CHECK:      scf.yield

// CHECK:      arith.muli
// CHECK:      arith.addi
// CHECK:    return

func.func @insert_cleartext_into_secret_dynamic(%arg0: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}, %insertIndex: index) -> (!secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) {
  %c7 = arith.constant 7 : i16
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) {
  ^body(%input0: tensor<8xi16>):
    %0 = tensor.insert %c7 into %input0[%insertIndex] {tensor_ext.layout = #layout} : tensor<8xi16>
    secret.yield %0 : tensor<8xi16>
  } -> (!secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout})
  return %0 : !secret.secret<tensor<8xi16>>
}

// CHECK:   func.func @insert_secret_into_secret(%[[tensor_secret:[^:]*]]: !secret.secret<tensor<1x16xi16>>
// CHECK-SAME:       %[[scalar_secret:[^:]*]]: !secret.secret<tensor<1x16xi16>>
// CHECK-DAG:    %[[c11:.*]] = arith.constant 11 : index
// CHECK-DAG:    %[[c0_index:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[c0_i16:.*]] = arith.constant 0 : i16
// CHECK-DAG:    %[[c3:.*]] = arith.constant 3 : index
// CHECK-DAG:    %[[c1:.*]] = arith.constant 1 : i16
// CHECK-DAG:    %[[dense1:.*]] = arith.constant dense<1> : tensor<1x16xi16>
// CHECK-DAG:    %[[dense0:.*]] = arith.constant dense<0> : tensor<1x16xi16>
// CHECK:    %[[res:.*]] = secret.generic(%[[tensor_secret]]: !secret.secret<tensor<1x16xi16>>, %[[scalar_secret]]: !secret.secret<tensor<1x16xi16>>) {
// CHECK:    ^body(%[[tensor_clear:.*]]: tensor<1x16xi16>, %[[scalar_clear:.*]]: tensor<1x16xi16>):

// This test differs from other tests in that the scalar mask has 1's and is then multiplied with the scalar_clear
// CHECK:      %[[scalar_mask1:.*]] = tensor.insert %[[c1]] into %[[dense0]][%[[c0_index]], %[[c3]]]
// CHECK:      tensor.insert %[[c1]] into %[[scalar_mask1]][%[[c0_index]], %[[c11]]]
// CHECK:      arith.muli
// CHECK:      arith.muli
// CHECK:      arith.addi
// CHECK:    return %[[res]]
func.func @insert_secret_into_secret(%arg0: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}, %arg1: !secret.secret<i16> {tensor_ext.layout = #scalar_layout}) -> (!secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) {
  %insertIndex = arith.constant 3 : index
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}, %arg1: !secret.secret<i16> {tensor_ext.layout = #scalar_layout}) {
  ^body(%t: tensor<8xi16>, %scalar: i16):
    %0 = tensor.insert %scalar into %t[%insertIndex] {tensor_ext.layout = #layout} : tensor<8xi16>
    secret.yield %0 : tensor<8xi16>
  } -> (!secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout})
  return %0 : !secret.secret<tensor<8xi16>>
}
