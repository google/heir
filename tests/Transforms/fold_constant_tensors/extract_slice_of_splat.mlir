// RUN: heir-opt --fold-constant-tensors %s | FileCheck %s

// CHECK: func @extract_slice_of_splat
func.func @extract_slice_of_splat(%arg0: i32) -> (tensor<1024xi32>) {
  // Fold a collapse shape of a constant
  // CHECK-NEXT: %[[splat:.+]] = tensor.splat
  // CHECK-NEXT: return %[[splat]]
  %0 = tensor.splat %arg0 : tensor<10x1024xi32>
  %slice = tensor.extract_slice %0[1, 0] [1, 1024] [1, 1] : tensor<10x1024xi32> to tensor<1024xi32>
  return %slice : tensor<1024xi32>
}
