// RUN: heir-opt %s --layout-propagation=min-slot-count=64 --convert-to-ciphertext-semantics=min-slot-count=64 | FileCheck %s

// Without coprime dimensions the matmul has no bicyclic kernel; the generic
// kernel lowers it with one ciphertext per row of the result, so the init is
// converted to that layout too.

// CHECK: func.func @matmul_non_coprime
// CHECK-NOT: linalg.matmul
// CHECK: return
// CHECK-SAME: tensor<4x64xf32>
func.func @matmul_non_coprime(%arg0: !secret.secret<tensor<4x6xf32>>) -> !secret.secret<tensor<4x9xf32>> {
  %weights = arith.constant dense<1.0> : tensor<6x9xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<4x9xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4x6xf32>>) {
  ^body(%input0: tensor<4x6xf32>):
    %1 = linalg.matmul ins(%input0, %weights : tensor<4x6xf32>, tensor<6x9xf32>) outs(%cst : tensor<4x9xf32>) -> tensor<4x9xf32>
    secret.yield %1 : tensor<4x9xf32>
  } -> !secret.secret<tensor<4x9xf32>>
  return %0 : !secret.secret<tensor<4x9xf32>>
}
