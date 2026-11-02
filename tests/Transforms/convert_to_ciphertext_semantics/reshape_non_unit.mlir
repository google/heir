// RUN: heir-opt %s --layout-propagation=min-slot-count=16 --convert-to-ciphertext-semantics=min-slot-count=16 | FileCheck %s

// Reshapes that split or merge dimensions of size other than 1 keep every
// element in its slot, so they lower to nothing.

// CHECK: func.func @reshape
// CHECK-NOT: tensor_ext.rotate
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<1x16xf32>
// CHECK-NOT: tensor_ext.rotate
// CHECK: return
func.func @reshape(%arg0: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>>) {
  ^body(%x: tensor<8xf32>):
    %expanded = tensor.expand_shape %x [[0, 1]] output_shape [2, 4] : tensor<8xf32> into tensor<2x4xf32>
    %sum = arith.addf %expanded, %expanded : tensor<2x4xf32>
    %collapsed = tensor.collapse_shape %sum [[0, 1]] : tensor<2x4xf32> into tensor<8xf32>
    secret.yield %collapsed : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
