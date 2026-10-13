// RUN: heir-opt --preprocessing-use-dynamic-resource-dir %s | FileCheck %s

// CHECK: func.func private @layout(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>,
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:         %[[WEIGHTS:.*]] = preprocessing.load_resource "weights.bin" from %[[DIR]] into %[[EMPTY]] : (!preprocessing.resource_dir, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         %[[SUM:.*]] = arith.addf %[[IN]], %[[WEIGHTS]] : tensor<4xf32>
// CHECK:         return %[[SUM]] : tensor<4xf32>
func.func private @layout(%input: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %weights = preprocessing.load_resource "weights.bin" into %empty : (tensor<4xf32>) -> tensor<4xf32>
  %sum = arith.addf %input, %weights : tensor<4xf32>
  return %sum : tensor<4xf32>
}

// CHECK: func.func @preprocess(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>,
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xf32>
// CHECK:         %[[RES:.*]] = call @layout(%[[IN]], %[[DIR]]) : (tensor<4xf32>, !preprocessing.resource_dir) -> tensor<4xf32>
// CHECK:         return %[[RES]] : tensor<4xf32>
func.func @preprocess(%input: tensor<4xf32>) -> tensor<4xf32> {
  %result = func.call @layout(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
