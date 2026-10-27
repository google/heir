// RUN: heir-opt --layout-propagation=min-slot-count=16 --layout-optimization=min-slot-count=16 --convert-to-ciphertext-semantics=min-slot-count=16 %s | FileCheck %s

// C[i, j] = sum_k A[i, k] * B[j, k]. Both operands are arguments, so the
// pulled-back layouts fold into them: 8 products, no rotations.

// CHECK: func.func @secret_matmul
// CHECK-NOT: tensor_ext.rotate
// CHECK-COUNT-8: arith.mulf
// CHECK-NOT: arith.mulf
// CHECK-NOT: tensor_ext.rotate
// CHECK: return
func.func @secret_matmul(%a: !secret.secret<tensor<4x8xf32>>, %b: !secret.secret<tensor<4x8xf32>>) -> !secret.secret<tensor<4x4xf32>> {
  %zero = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<4x8xf32>>, %b: !secret.secret<tensor<4x8xf32>>) {
  ^body(%x: tensor<4x8xf32>, %y: tensor<4x8xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %y : tensor<4x8xf32>, tensor<4x8xf32>) outs(%zero : tensor<4x4xf32>) {
    ^bb0(%l: f32, %r: f32, %acc: f32):
      %2 = arith.mulf %l, %r : f32
      %3 = arith.addf %acc, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<4x4xf32>
    secret.yield %1 : tensor<4x4xf32>
  } -> !secret.secret<tensor<4x4xf32>>
  return %0 : !secret.secret<tensor<4x4xf32>>
}
