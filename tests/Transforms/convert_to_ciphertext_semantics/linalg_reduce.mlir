// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

// CHECK-LABEL: @convert_linalg_reduce
#map = affine_map<(d0, d1) -> (d0 * 32 + d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 * 32)>

// FIXME: decide what this pass should do
func.func @convert_linalg_reduce(
    %arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>},
    %arg1: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>}) ->
       (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0) -> (d0)>}) {
  %cst = arith.constant dense<0> : tensor<32xi16>
  %cst_0 = arith.constant dense<0> : tensor<32xi16>

  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32x32xi16>>, !secret.secret<tensor<32x32xi16>>)
                      attrs = {arg0 = {layout = #map}, arg1 = {layout = #map}, layout = [#map1]} {
  ^body(%input0: tensor<32x32xi16>, %input1: tensor<32x32xi16>):
    %1 = tensor_ext.assign_layout %cst {layout = #map1} : tensor<32xi16>

    %reduced = linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} }
      ins(%input0 : tensor<32x32xi16>)
      outs(%1 : tensor<32xi16>)
      dimensions = [0]  {layout = [#map1]}

    %2 = tensor_ext.assign_layout %cst_0 {layout = #map1} : tensor<32xi16>
    %3 = tensor_ext.convert_layout %2 {from_layout = #map1, layout = [#map2], to_layout = #map2} : tensor<32xi16>

    %reduced_1 = linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} }
      ins(%input1 : tensor<32x32xi16>)
      outs(%3 : tensor<32xi16>)
      dimensions = [1]  {layout = [#map2]}

    %4 = tensor_ext.convert_layout %reduced_1 {from_layout = #map2, layout = [#map1], to_layout = #map1} : tensor<32xi16>
    %5 = arith.addi %reduced, %4 {layout = [#map1]} : tensor<32xi16>
    secret.yield %5 : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return %0 : !secret.secret<tensor<32xi16>>
}
