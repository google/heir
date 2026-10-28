// RUN: heir-opt --layout-optimization --canonicalize %s | FileCheck %s

// The hoisted conversion of the collapse's operand is costed in the operand's
// (uncollapsed) layout. The operand is computed, so its layout is not free and
// the convert_layout stays.

#in = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i1 = 0 and ct = i0 and slot = 8i2 + i3 and 0 <= i0 <= 1 and 0 <= i2 <= 3 and 0 <= i3 <= 7 and 0 <= slot <= 31 }">
#sum = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and slot = 8i1 + i2 and 0 <= i1 <= 3 and 0 <= i2 <= 7 and 0 <= slot <= 31 }">
#flat = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 8i0 + i1 and 0 <= i0 <= 3 and 0 <= i1 <= 7 and 0 <= slot <= 31 }">
#transposed = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 4i1 + i0 and 0 <= i0 <= 3 and 0 <= i1 <= 7 and 0 <= slot <= 31 }">

// CHECK: func.func @collapse_computed
// CHECK: linalg.reduce
// CHECK-NEXT: tensor.collapse_shape
// CHECK-NEXT: tensor_ext.convert_layout
// CHECK-NEXT: secret.yield
func.func @collapse_computed(%arg0: !secret.secret<tensor<2x1x4x8xf32>> {tensor_ext.layout = #in}) -> (!secret.secret<tensor<4x8xf32>> {tensor_ext.layout = #transposed}) {
  %zero = arith.constant dense<0.0> : tensor<1x4x8xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<2x1x4x8xf32>> {tensor_ext.layout = #in}) {
  ^body(%input0: tensor<2x1x4x8xf32>):
    %init = tensor_ext.assign_layout %zero <layout = #sum> {tensor_ext.layout = #sum} : tensor<1x4x8xf32>
    %sum = linalg.reduce { arith.addf {tensor_ext.layout = #sum} } ins(%input0 : tensor<2x1x4x8xf32>) outs(%init : tensor<1x4x8xf32>) dimensions = [0] {tensor_ext.layout = #sum}
    %collapsed = tensor.collapse_shape %sum [[0, 1], [2]] {tensor_ext.layout = #flat} : tensor<1x4x8xf32> into tensor<4x8xf32>
    %1 = tensor_ext.convert_layout %collapsed <from_layout = #flat, to_layout = #transposed> {tensor_ext.layout = #transposed} : tensor<4x8xf32>
    secret.yield %1 : tensor<4x8xf32>
  } -> (!secret.secret<tensor<4x8xf32>> {tensor_ext.layout = #transposed})
  return %0 : !secret.secret<tensor<4x8xf32>>
}
