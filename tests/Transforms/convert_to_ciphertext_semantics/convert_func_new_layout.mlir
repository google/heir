// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

// CHECK-DAG: [[layout:[^ ]*]] = #tensor_ext.new_layout
// CHECK-DAG: [[orig_type:[^ ]*]] = #tensor_ext.original_type<originalType = tensor<32x32xi16>, layout = [[layout]]>

// CHECK: @minimal_example(
// CHECK-SAME: [[arg0:%[^:]*]]: !secret.secret<tensor<1x1024xi16>>
// CHECK-SAME: {tensor_ext.original_type = [[orig_type]]}
// CHECK-SAME: -> (!secret.secret<tensor<1x1024xi16>> {tensor_ext.original_type = [[orig_type]]})
#new_layout = #tensor_ext.new_layout<domainSize=2, localSize=4, relation="(d0, d1, d2, d3, d4, d5, d6, d7) : (d0 * 32 + d1 - d4 * 1024 - d5 == 0, -d2 + d4 == 0, d3 - d6 * 1024 - d7 == 0, d5 - d7 == 0, d1 - d4 * 1024 - d5 + 992 >= 0, -d1 + d4 * 1024 + d5 >= 0, -d1 + 31 >= 0, d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 1023 >= 0, -d5 + 1023 >= 0, d5 >= 0, -d3 + d6 * 1024 + 1023 >= 0, d3 - d6 * 1024 >= 0)">
module {
  func.func @minimal_example(%arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #new_layout}, %arg1: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #new_layout}) -> (!secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #new_layout}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #new_layout}, %arg1: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #new_layout}) {
    ^body(%input0: tensor<32x32xi16>, %input1: tensor<32x32xi16>):
      %1 = arith.addi %input0, %input1 {tensor_ext.layout = #new_layout} : tensor<32x32xi16>
      secret.yield %1 : tensor<32x32xi16>
    } -> (!secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #new_layout})
    return %0 : !secret.secret<tensor<32x32xi16>>
  }
}
