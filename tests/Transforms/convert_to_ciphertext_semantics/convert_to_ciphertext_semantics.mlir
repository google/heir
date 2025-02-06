// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

#map = affine_map<(d0, d1) -> (d0 * 32 + d1)>

// CHECK-LABEL: @convert_minimal_example(
// CHECK-SAME: [[arg0:%[^:]*]]: !secret.secret<tensor<1024xi16>>
// CHECK-SAME: {tensor_ext.original_type = #tensor_ext.original_type<originalType = !secret.secret<tensor<32x32xi16>>, layout = (d0, d1) -> (d0 * 32 + d1)>
// CHECK-SAME: -> (!secret.secret<tensor<1024xi16>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = !secret.secret<tensor<32x32xi16>>, layout = (d0, d1) -> (d0 * 32 + d1)>})
func.func @convert_minimal_example(
    %arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = affine_map<(d0, d1) -> (d0 * 32 + d1)>}) ->
       (!secret.secret<tensor<32x32xi16>> {tensor_ext.layout = affine_map<(d0, d1) -> (d0 * 32 + d1)>}) {
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<32x32xi16>>)
                      attrs = {
                        __argattrs=[{tensor_ext.layout = #map}],
                        __resattrs=[{tensor_ext.layout = #map}]
                      } {
  ^body(%input0: tensor<32x32xi16>):
    %1 = arith.addi %input0, %input0 {tensor_ext.layout = [#map]} : tensor<32x32xi16>
    secret.yield %1 : tensor<32x32xi16>
  } -> !secret.secret<tensor<32x32xi16>>
  return %0 : !secret.secret<tensor<32x32xi16>>
}
