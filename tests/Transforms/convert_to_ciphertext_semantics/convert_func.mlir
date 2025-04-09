// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

#row_major = #tensor_ext.layout<map = (d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG: [[layout:[^ ]*]] = #tensor_ext.layout<map = (d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG: [[orig_type:[^ ]*]] = #tensor_ext.original_type<originalType = tensor<32x32xi16>, layout = [[layout]]>

// CHECK: @convert_minimal_example(
// CHECK-SAME: [[arg0:%[^:]*]]: !secret.secret<tensor<1024xi16>>
// CHECK-SAME: {tensor_ext.original_type = [[orig_type]]}
// CHECK-SAME: -> (!secret.secret<tensor<1024xi16>> {tensor_ext.original_type = [[orig_type]]})
func.func @convert_minimal_example(
    %arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #row_major}) ->
       (!secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #row_major}) {
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<32x32xi16>>)
                      attrs = {
                        __argattrs=[{tensor_ext.layout = #row_major}],
                        __resattrs=[{tensor_ext.layout = #row_major}]
                      } {
  ^body(%input0: tensor<32x32xi16>):
    %1 = arith.addi %input0, %input0 {tensor_ext.layout = #row_major} : tensor<32x32xi16>
    secret.yield %1 : tensor<32x32xi16>
  } -> !secret.secret<tensor<32x32xi16>>
  return %0 : !secret.secret<tensor<32x32xi16>>
}
