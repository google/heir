// RUN: heir-opt --layout-propagation=min-slot-count=32 %s | FileCheck %s

// The reduce kernel adds its init to the result. The input arrives
// column-major and is converted to row-major for the kernel, so the init is
// laid out like the row-major sum.

#col_major = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 4i1 + i0 and 0 <= i0 <= 3 and 0 <= i1 <= 7 and 0 <= slot <= 31 }">

// CHECK: #[[$SUM:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-8i0 + slot) mod 32 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 31 }">
// CHECK: func.func @row_sum
// CHECK: %[[init:.*]] = tensor_ext.assign_layout %{{.*}} <layout = #[[$SUM]]>
// CHECK: linalg.reduce {{.*}}outs(%[[init]] : tensor<4xf32>) {{.*}}tensor_ext.layout = #[[$SUM]]
func.func @row_sum(%x: !secret.secret<tensor<4x8xf32>> {tensor_ext.layout = #col_major}) -> !secret.secret<tensor<4xf32>> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<4x8xf32>>) {
  ^body(%input: tensor<4x8xf32>):
    %sum = linalg.reduce { arith.addf } ins(%input : tensor<4x8xf32>) outs(%zero : tensor<4xf32>) dimensions = [1]
    secret.yield %sum : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
