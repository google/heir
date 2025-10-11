// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

// CHECK-DAG: [[layout:[^ ]*]] = #tensor_ext.layout
// CHECK-DAG: [[orig_type:[^ ]*]] = #tensor_ext.original_type<originalType = tensor<32x32xi16>, layout = [[layout]]>

// CHECK: @minimal_example(
// CHECK-SAME: [[arg0:%[^:]*]]: !secret.secret<tensor<1x1024xi16>>
// CHECK-SAME: {tensor_ext.original_type = [[orig_type]]}
// CHECK-SAME: -> (!secret.secret<tensor<1x1024xi16>> {tensor_ext.original_type = [[orig_type]]})
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-32i0 - i1 + slot) mod 1024 = 0 and 0 <= i0 <= 31 and 0 <= i1 <= 31 and 0 <= slot <= 1023 }">
module {
  func.func @minimal_example(%arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #layout}, %arg1: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #layout}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #layout}, %arg1: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #layout}) {
    ^body(%input0: tensor<32x32xi16>, %input1: tensor<32x32xi16>):
      %1 = arith.addi %input0, %input1 {tensor_ext.layout = #layout} : tensor<32x32xi16>
      secret.yield %1 : tensor<32x32xi16>
    } -> (!secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<32x32xi16>>
  }
}
