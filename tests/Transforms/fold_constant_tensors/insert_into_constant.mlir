// RUN: heir-opt --fold-constant-tensors %s | FileCheck %s

// CHECK: func @canonicalize_insert_after_constant
func.func @canonicalize_insert_after_constant() -> (tensor<2x2xi32>) {
  // Fold an insert into a splat.
  // CHECK: %[[C4:.+]] = arith.constant dense<{{\[\[}}1, 2], [4, 4]]> : tensor<2x2xi32>
  // CHECK-NEXT: return %[[C4]]
  %cst = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4_i32 = arith.constant 4 : i32
  %inserted = tensor.insert %c4_i32 into %cst[%c1, %c0] : tensor<2x2xi32>
  return %inserted : tensor<2x2xi32>
}
