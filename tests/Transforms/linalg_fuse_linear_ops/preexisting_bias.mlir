// RUN: heir-opt --linalg-fuse-linear-ops %s | FileCheck %s

// Tests that fusing a multiplication (scale) into a linalg.matmul updates the
// pre-existing outs accumulator appropriately.

// CHECK: func.func @matmul_preexisting_bias
// CHECK-SAME: %[[arg1:[a-zA-Z0-9_]+]]: tensor<3x4xf32>
// CHECK-SAME: %[[arg3:[a-zA-Z0-9_]+]]: tensor<4xf32>
// CHECK: %[[EMPTY_W:.*]] = tensor.empty() : tensor<3x4xf32>
// CHECK: %[[BROADCAST_W:.*]] = linalg.broadcast ins(%[[arg3]] : tensor<4xf32>) outs(%[[EMPTY_W]] : tensor<3x4xf32>) dimensions = [0]
// CHECK: %[[SCALED_W:.*]] = arith.mulf %[[arg1]], %[[BROADCAST_W]] : tensor<3x4xf32>
// CHECK: %[[RESULT:.*]] = linalg.matmul ins(%arg0, %[[SCALED_W]] : tensor<2x3xf32>, tensor<3x4xf32>) outs(%arg2 : tensor<2x4xf32>)
// CHECK: return %[[RESULT]]
func.func @matmul_preexisting_bias(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x4xf32>, %arg3: tensor<4xf32>) -> tensor<2x4xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%arg2 : tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = tensor.empty() : tensor<2x4xf32>
  %broadcasted = linalg.broadcast ins(%arg3 : tensor<4xf32>) outs(%1 : tensor<2x4xf32>) dimensions = [0]
  %2 = arith.mulf %0, %broadcasted : tensor<2x4xf32>
  return %2 : tensor<2x4xf32>
}
