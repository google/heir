// RUN: heir-opt --implement-shift-network --canonicalize %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (2i2 - i3 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0, e1, e2: i0 = 0 and ct = 0 and 16e2 = -i1 + slot + 16e0 and 0 <= i1 <= 1023 and 0 <= slot <= 1023 and 0 <= e1 <= 3 and -3 + i1 - 16e0 <= 4e1 <= i1 - 16e0) }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x2x2x2xf32>, layout = #layout>
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<1x1024xf32>>
  // CHECK: return %[[arg0]]
  func.func @main(%arg0: !secret.secret<tensor<1x1024xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x4x4xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 1023 }">>}, %arg1: tensor<2x1x3x3xf32>) -> (!secret.secret<tensor<1x1024xf32>> {tensor_ext.original_type = #original_type}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = tensor_ext.remap %input0 {permutation = #layout1} : tensor<1x1024xf32>
      secret.yield %1 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
