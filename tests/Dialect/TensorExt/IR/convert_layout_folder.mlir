// RUN: heir-opt --canonicalize --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @fold
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: return
func.func @fold(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}) {
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<32xi16>>) attrs = {arg0 = {layout = #map}, layout = [#map]} {
  ^body(%input0: tensor<32xi16>):
    %1 = tensor_ext.convert_layout %input0 {from_layout = #map, layout = [#map1], to_layout = #map1} : tensor<32xi16>
    %2 = tensor_ext.convert_layout %1 {from_layout = #map1, layout = [#map], to_layout = #map} : tensor<32xi16>
    secret.yield %2 : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return %0 : !secret.secret<tensor<32xi16>>
}

// -----

#map = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: func @noop
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: return
func.func @noop(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map}) {
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<32xi16>>) attrs = {arg0 = {layout = #map}, layout = [#map]} {
  ^body(%input0: tensor<32xi16>):
    %1 = tensor_ext.convert_layout %input0 {from_layout = #map, layout = [#map], to_layout = #map} : tensor<32xi16>
    secret.yield %1 : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return %0 : !secret.secret<tensor<32xi16>>
}
