// RUN: heir-opt --fold-constant-tensors --split-input-file %s | FileCheck %s

// CHECK: func @extract
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x3xi32>)
func.func @extract(%arg0: tensor<1x3xi32>) -> (i32) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[V0:.+]] = tensor.extract %[[arg0]][%[[C0]], %[[C0]]]
  // CHECK-NEXT: return %[[V0]]
  %c0 = arith.constant 0 : index
  %slice = tensor.extract_slice %arg0[0, 0][1, 3][1, 1] : tensor<1x3xi32> to tensor<3xi32>
  %extracted = tensor.extract %slice[%c0] : tensor<3xi32>
  return %extracted : i32
}

// -----

// CHECK: func @index
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x3xi32>)
func.func @index(%arg0: tensor<2x3xi32>) -> (i32) {
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: %[[V0:.+]] = tensor.extract %[[arg0]][%[[C1]], %[[C1]]]
  // CHECK-NEXT: return %[[V0]]
  %c1 = arith.constant 1 : index
  %slice = tensor.extract_slice %arg0[1, 0][1, 3][1, 1] : tensor<2x3xi32> to tensor<3xi32>
  %extracted = tensor.extract %slice[%c1] : tensor<3xi32>
  return %extracted : i32
}
