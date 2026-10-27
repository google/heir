// RUN: heir-opt %s --layout-propagation=min-slot-count=16 --convert-to-ciphertext-semantics=min-slot-count=16 | FileCheck %s

// Rotations act within one ciphertext, so the rotate-and-reduce kernel
// rejects an input packed into two and the generic kernel sums each row in its
// own ciphertext.

// CHECK: func.func @row_sum
// CHECK-NOT: linalg.reduce
// CHECK: tensor.extract_slice
// CHECK: tensor_ext.rotate
// CHECK: tensor.extract_slice
// CHECK: tensor_ext.rotate
// CHECK: return
func.func @row_sum(%x: !secret.secret<tensor<4x8xf32>>) -> !secret.secret<tensor<4xf32>> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<4x8xf32>>) {
  ^body(%input: tensor<4x8xf32>):
    %sum = linalg.reduce { arith.addf } ins(%input : tensor<4x8xf32>) outs(%zero : tensor<4xf32>) dimensions = [1]
    secret.yield %sum : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
