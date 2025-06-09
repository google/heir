// RUN: heir-opt --fold-constant-tensors %s | FileCheck %s

// CHECK: func @collapse_empty
func.func @collapse_empty() -> (tensor<2x2xi32>) {
  // Fold a collapse shape of an empty tensor.
  // CHECK: %[[C4:.+]] = tensor.empty() : tensor<2x2xi32>
  // CHECK-NEXT: return %[[C4]]
  %cst = tensor.empty() : tensor<1x2x2xi32>
  %collapsed = tensor.collapse_shape %cst [[0, 1], [2]] : tensor<1x2x2xi32> into tensor<2x2xi32>
  return %collapsed : tensor<2x2xi32>
}
