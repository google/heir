// RUN: heir-opt --forward-insert-to-extract %s | FileCheck %s

// CHECK-LABEL: func.func @extract_inserted_value
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
func.func @extract_inserted_value(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<2xi32>
  %inserted = tensor.insert %arg0 into %empty[%c0] : tensor<2xi32>
  %inserted_1 = tensor.insert %arg1 into %inserted[%c1] : tensor<2xi32>
  %result = tensor.extract %inserted_1[%c0] : tensor<2xi32>
  // CHECK: return %[[ARG0]] : i32
  return %result : i32
}
