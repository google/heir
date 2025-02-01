// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

// CHECK-LABEL: @convert_minimal_example
#map = affine_map<(d0, d1) -> (d0 * 32 + d1)>

func.func @convert_minimal_example(
    %arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>}) ->
       (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>}) {
  %0 = secret.generic ins(%arg0, : !secret.secret<tensor<32x32xi16>>)
                      attrs = {arg0 = {layout = #map}, layout = [#map]} {
  ^body(%input0: tensor<32x32xi16>):
    %1 = linalg.concat dim(0) %input0, %input0 : (tensor<32x32xi16>, tensor<32x32xi16>) -> tensor<64x32xi16>
    secret.yield %1 : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return %0 : !secret.secret<tensor<32xi16>>
}
