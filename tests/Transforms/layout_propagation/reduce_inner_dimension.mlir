// RUN: heir-opt --layout-propagation=min-slot-count=32 %s | FileCheck %s

// The sum of each row of a row-major 4x8 matrix is only in the slot of the
// row's first element, and so is the row mean the broadcast reads.

// CHECK: #[[$SUM:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-8i0 + slot) mod 32 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 31 }">
// CHECK: func.func @row_center
// CHECK: linalg.reduce {{.*}}tensor_ext.layout = #[[$SUM]]
// CHECK: arith.divf {{.*}}tensor_ext.layout = #[[$SUM]]
// CHECK: linalg.broadcast
func.func @row_center(%x: !secret.secret<tensor<4x8xf32>>) -> !secret.secret<tensor<4x8xf32>> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %eight = arith.constant dense<8.0> : tensor<4xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<4x8xf32>>) {
  ^body(%input: tensor<4x8xf32>):
    %sum = linalg.reduce { arith.addf } ins(%input : tensor<4x8xf32>) outs(%zero : tensor<4xf32>) dimensions = [1]
    %mean = arith.divf %sum, %eight : tensor<4xf32>
    %wide = linalg.broadcast ins(%mean : tensor<4xf32>) outs(%empty : tensor<4x8xf32>) dimensions = [1]
    %centered = arith.subf %input, %wide : tensor<4x8xf32>
    secret.yield %centered : tensor<4x8xf32>
  } -> !secret.secret<tensor<4x8xf32>>
  return %0 : !secret.secret<tensor<4x8xf32>>
}
