// RUN: heir-opt --layout-propagation %s | FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK: #map1 = affine_map<(d0) -> (d0)>
// CHECK: #map2 = affine_map<(d0) -> (d0 * 32)>
// CHECK: func.func @insert_conversion
// CHECK-SAME: (%arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>})
// CHECK-SAME: (%arg1: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>})
// CHECK-SAME: -> !secret.secret<tensor<32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0) -> (d0)>}
// CHECK: %[[GENERIC:.*]] = secret.generic(%arg0, %arg1 : !secret.secret<tensor<32x32xi16>>, !secret.secret<tensor<32x32xi16>>)
// CHECK-SAME: attrs = {arg0 = {tensor_ext.layout = #map}, arg1 = {tensor_ext.layout = #map}, layout = [#map1]}
// CHECK: ^body(%[[INPUT0:.*]]: tensor<32x32xi16>, %[[INPUT1:.*]]: tensor<32x32xi16>):
// CHECK:   %[[ASSIGN:.*]] = tensor_ext.assign_layout %{{.*}} {tensor_ext.layout = #map1} : tensor<32xi16>
// CHECK:   %[[REDUCE:.*]] = linalg.reduce
// CHECK-SAME: ins(%[[INPUT0]] : tensor<32x32xi16>) outs(%[[ASSIGN]] : tensor<32xi16>)
// CHECK-SAME: {tensor_ext.layout = [#map1]}
// CHECK:   %[[ASSIGN2:.*]] = tensor_ext.assign_layout %{{.*}} {tensor_ext.layout = #map1} : tensor<32xi16>
// CHECK:   %[[CONVERT:.*]] = tensor_ext.convert_layout %[[ASSIGN2]] {from_layout = #map1, layout = [#map2], to_layout = #map2} : tensor<32xi16>
// CHECK:   %[[REDUCE2:.*]] = linalg.reduce
// CHECK-SAME: ins(%[[INPUT1]] : tensor<32x32xi16>) outs(%[[CONVERT]] : tensor<32xi16>)
// CHECK-SAME: {tensor_ext.layout = [#map2]}
// CHECK:   %[[CONVERT2:.*]] = tensor_ext.convert_layout %[[REDUCE2]] {from_layout = #map2, layout = [#map1], to_layout = #map1} : tensor<32xi16>
// CHECK:   %[[ADD:.*]] = arith.addi %[[REDUCE]], %[[CONVERT2]] {tensor_ext.layout = [#map1]} : tensor<32xi16>
// CHECK:   secret.yield %[[ADD]] : tensor<32xi16>
// CHECK: return %[[GENERIC]]

!tensor = tensor<32x32xi16>
!tensor2 = tensor<32xi16>
!stensor = !secret.secret<!tensor>
!stensor2 = !secret.secret<!tensor2>

func.func @insert_conversion(%arg0: !stensor, %arg1: !stensor) -> !stensor2 {
  %out_1 = arith.constant dense<0> : !tensor2
  %out_2 = arith.constant dense<0> : !tensor2

  %0 = secret.generic(%arg0, %arg1: !stensor, !stensor) {
  ^body(%pt_arg0: !tensor, %pt_arg1: !tensor):
    %1 = linalg.reduce { arith.addi } ins(%pt_arg0:!tensor) outs(%out_1:!tensor2) dimensions = [0]
    %2 = linalg.reduce { arith.addi } ins(%pt_arg1:!tensor) outs(%out_2:!tensor2) dimensions = [1]
    %3 = arith.addi %1, %2 : !tensor2
    secret.yield %3 : !tensor2
  } -> !stensor2
  return %0 : !stensor2
}
