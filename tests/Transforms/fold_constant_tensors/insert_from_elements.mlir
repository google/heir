// RUN: heir-opt --fold-constant-tensors --split-input-file %s | FileCheck %s

// CHECK: func @full_inserts
// CHECK-SAME: (%[[arg0:.*]]: i32, %[[arg1:.*]]: i32, %[[arg2:.*]]: i32)
func.func @full_inserts(%arg0: i32, %arg1: i32, %arg2: i32) -> (tensor<3xi32>) {
  // Fold inserts into a from_elements op
  // CHECK: %[[C4:.+]] = tensor.from_elements %[[arg0]], %[[arg1]], %[[arg2]]
  // CHECK-NEXT: return %[[C4]]
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %inserted = tensor.insert %arg0 into %cst[%c0] : tensor<3xi32>
  %inserted_1 = tensor.insert %arg1 into %inserted[%c1] : tensor<3xi32>
  %inserted_2 = tensor.insert %arg2 into %inserted_1[%c2] : tensor<3xi32>
  return %inserted_2 : tensor<3xi32>
}

// -----

// CHECK: func @incomplete
// CHECK-SAME: (%[[arg0:.*]]: i32, %[[arg1:.*]]: i32, %[[arg2:.*]]: i32)
func.func @incomplete(%arg0: i32, %arg1: i32, %arg2: i32) -> (tensor<3xi32>) {
  // Fold inserts into a from_elements op
  // CHECK: %[[C3:.*]] = arith.constant 3 : i32
  // CHECK: %[[C4:.+]] = tensor.from_elements %[[arg0]], %[[arg1]], %[[C3]]
  // CHECK-NEXT: return %[[C4]]
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %inserted = tensor.insert %arg0 into %cst[%c0] : tensor<3xi32>
  %inserted_1 = tensor.insert %arg1 into %inserted[%c1] : tensor<3xi32>
  return %inserted_1 : tensor<3xi32>
}

// -----

// CHECK: func @double_insert
// CHECK-SAME: (%[[arg0:.*]]: i32, %[[arg1:.*]]: i32, %[[arg2:.*]]: i32)
func.func @double_insert(%arg0: i32, %arg1: i32, %arg2: i32) -> (tensor<3xi32>) {
  // Folding fails from inserting two different values into the same index.
  // CHECK-COUNT-3: tensor.insert
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %inserted = tensor.insert %arg0 into %cst[%c0] : tensor<3xi32>
  %inserted_1 = tensor.insert %arg1 into %inserted[%c1] : tensor<3xi32>
  %inserted_2 = tensor.insert %arg2 into %inserted_1[%c1] : tensor<3xi32>
  return %inserted_2 : tensor<3xi32>
}
