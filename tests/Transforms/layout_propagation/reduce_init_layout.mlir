// RUN: heir-opt --layout-propagation=min-slot-count=32 %s | FileCheck %s

// The reduce kernel adds its init to the result, so an init laid out
// differently (here one element per ciphertext) is converted to the result
// layout.

#per_ct = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = i0 and slot = 0 and 0 <= i0 <= 3 }">

// CHECK: #[[$SUM:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-8i0 + slot) mod 32 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 31 }">
// CHECK: func.func @row_sum
// CHECK: %[[init:.*]] = tensor_ext.convert_layout {{.*}}to_layout = #[[$SUM]]
// CHECK: linalg.reduce {{.*}}outs(%[[init]] : tensor<4xf32>)
func.func @row_sum(%x: !secret.secret<tensor<4x8xf32>>) -> !secret.secret<tensor<4xf32>> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<4x8xf32>>) {
  ^body(%input: tensor<4x8xf32>):
    %init = tensor_ext.assign_layout %zero <layout = #per_ct> {tensor_ext.layout = #per_ct} : tensor<4xf32>
    %sum = linalg.reduce { arith.addf } ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
    secret.yield %sum : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
