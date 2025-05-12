// RUN: heir-opt --fold-constant-tensors --split-input-file %s | FileCheck %s

// CHECK: func.func @insert(%[[input0:.*]]: i32)
// CHECK: %[[v0:.*]] = tensor.from_elements %[[input0]], %[[input0]]
// CHECK: return %[[v0]] : tensor<2xi32>
module {
  func.func @insert(%arg0: i32) -> tensor<2xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<1> : tensor<2xi32>
    %inserted = tensor.insert %arg0 into %cst[%c0] : tensor<2xi32>
    %inserted_0 = tensor.insert %arg0 into %inserted[%c1] : tensor<2xi32>
    return %inserted_0 : tensor<2xi32>
  }
}

// -----

// CHECK: func.func @multidim(%[[input0:.*]]: i32)
// CHECK: %[[v0:.*]] = tensor.from_elements %[[input0]], %[[input0]]
// CHECK: return %[[v0]] : tensor<1x2xi32>
module {
  func.func @multidim(%arg0: i32) -> tensor<1x2xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<1> : tensor<1x2xi32>
    %inserted = tensor.insert %arg0 into %cst[%c0, %c0] : tensor<1x2xi32>
    %inserted_0 = tensor.insert %arg0 into %inserted[%c0, %c1] : tensor<1x2xi32>
    return %inserted_0 : tensor<1x2xi32>
  }
}

// -----

// Overriding a value.

// CHECK: func.func @override_value
// CHECK-SAME: (%[[input0:.*]]: i32, %[[input1:.*]]: i32)
// CHECK: %[[c1:.*]] = arith.constant 1 : i32
// CHECK: %[[v0:.*]] = tensor.from_elements %[[input1]], %[[c1]]
// CHECK: return
module {
  func.func @override_value(%arg0: i32, %arg1: i32) -> tensor<1x2xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<1> : tensor<1x2xi32>
    %inserted = tensor.insert %arg0 into %cst[%c0, %c0] : tensor<1x2xi32>
    %inserted_0 = tensor.insert %arg1 into %inserted[%c0, %c0] : tensor<1x2xi32>
    return %inserted_0 : tensor<1x2xi32>
  }
}

// -----

// Intermediate use of a value.

// CHECK: func.func @intermediate_use
// CHECK-COUNT-2: tensor.insert
// CHECK: return
module {
  func.func @intermediate_use(%arg0: i32, %arg1: i32) -> (tensor<1x2xi32>, tensor<1x2xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<1> : tensor<1x2xi32>
    %inserted = tensor.insert %arg0 into %cst[%c0, %c0] : tensor<1x2xi32>
    %v0 = arith.addi %inserted, %inserted: tensor<1x2xi32>
    %inserted_0 = tensor.insert %arg1 into %inserted[%c0, %c1] : tensor<1x2xi32>
    return %v0, %inserted_0 : tensor<1x2xi32>, tensor<1x2xi32>
  }
}

// -----

// Indeterminate indices.

// CHECK: func.func @indeterminate_indices
// CHECK-COUNT-2: tensor.insert
// CHECK: return
module {
  func.func @indeterminate_indices(%arg0: i32, %arg1: index) -> (tensor<1x2xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<1> : tensor<1x2xi32>
    %inserted = tensor.insert %arg0 into %cst[%c0, %arg1] : tensor<1x2xi32>
    %inserted_0 = tensor.insert %arg0 into %inserted[%c0, %c1] : tensor<1x2xi32>
    return %inserted_0 : tensor<1x2xi32>
  }
}
