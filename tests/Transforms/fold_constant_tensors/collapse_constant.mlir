// RUN: heir-opt --fold-constant-tensors %s | FileCheck %s

// CHECK: func @collapse_constant
func.func @collapse_constant() -> (tensor<2x2xi32>) {
  // Fold a collapse shape of a constant
  // CHECK: %[[C4:.+]] = arith.constant dense<{{\[\[}}1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK-NEXT: return %[[C4]]
  %cst = arith.constant dense<[[[1, 2], [3, 4]]]> : tensor<1x2x2xi32>
  %collapsed = tensor.collapse_shape %cst [[0, 1], [2]] : tensor<1x2x2xi32> into tensor<2x2xi32>
  return %collapsed : tensor<2x2xi32>
}
