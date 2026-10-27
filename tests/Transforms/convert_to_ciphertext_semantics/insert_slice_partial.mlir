// RUN: heir-opt --layout-propagation=min-slot-count=16 --convert-to-ciphertext-semantics=min-slot-count=16 %s | FileCheck %s

// A concatenation of two slices along a non-unit dimension: each slice is
// moved into its part of the result layout and masked in.

// CHECK: func.func @concat
// CHECK-NOT: tensor.insert_slice
// CHECK-COUNT-2: tensor_ext.remap
// CHECK: return
func.func @concat(%a: !secret.secret<tensor<1x4xf32>>, %b: !secret.secret<tensor<1x4xf32>>) -> !secret.secret<tensor<1x8xf32>> {
  %empty = tensor.empty() : tensor<1x8xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<1x4xf32>>, %b: !secret.secret<tensor<1x4xf32>>) {
  ^body(%x: tensor<1x4xf32>, %y: tensor<1x4xf32>):
    %1 = tensor.insert_slice %x into %empty[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x8xf32>
    %2 = tensor.insert_slice %y into %1[0, 4] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x8xf32>
    secret.yield %2 : tensor<1x8xf32>
  } -> !secret.secret<tensor<1x8xf32>>
  return %0 : !secret.secret<tensor<1x8xf32>>
}

// The slices replicate with period 8 and the result with period 16, so the
// slices occupy slots the result leaves empty; they are still moved into the
// result layout.

// CHECK: func.func @concat_replicated
// CHECK-NOT: tensor.insert_slice
// CHECK-COUNT-2: tensor_ext.remap
// CHECK: return
func.func @concat_replicated(%a: !secret.secret<tensor<1x6xf32>>, %b: !secret.secret<tensor<1x6xf32>>) -> !secret.secret<tensor<1x12xf32>> {
  %empty = tensor.empty() : tensor<1x12xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<1x6xf32>>, %b: !secret.secret<tensor<1x6xf32>>) {
  ^body(%x: tensor<1x6xf32>, %y: tensor<1x6xf32>):
    %1 = tensor.insert_slice %x into %empty[0, 0] [1, 6] [1, 1] : tensor<1x6xf32> into tensor<1x12xf32>
    %2 = tensor.insert_slice %y into %1[0, 6] [1, 6] [1, 1] : tensor<1x6xf32> into tensor<1x12xf32>
    secret.yield %2 : tensor<1x12xf32>
  } -> !secret.secret<tensor<1x12xf32>>
  return %0 : !secret.secret<tensor<1x12xf32>>
}

// The slice fills slots 8-15 of the first destination ciphertext and slots 0-7
// of the second, so it is moved into two ciphertexts, one per destination
// ciphertext.

// CHECK: func.func @slice_across_ciphertexts
// CHECK: tensor_ext.remap
// CHECK-SAME: tensor<2x16xf32>
// CHECK-DAG: tensor.extract_slice %[[moved:.*]][0, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
// CHECK-DAG: tensor.extract_slice %[[moved]][1, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
// CHECK: return
func.func @slice_across_ciphertexts(%a: !secret.secret<tensor<16xf32>>, %b: !secret.secret<tensor<32xf32>>) -> !secret.secret<tensor<32xf32>> {
  %0 = secret.generic(%a: !secret.secret<tensor<16xf32>>, %b: !secret.secret<tensor<32xf32>>) {
  ^body(%x: tensor<16xf32>, %y: tensor<32xf32>):
    %1 = tensor.insert_slice %x into %y[8] [16] [1] : tensor<16xf32> into tensor<32xf32>
    secret.yield %1 : tensor<32xf32>
  } -> !secret.secret<tensor<32xf32>>
  return %0 : !secret.secret<tensor<32xf32>>
}
