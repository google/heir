// RUN: heir-opt --linalg-fuse-linear-ops --split-input-file %s | FileCheck %s

// CHECK: func.func @fuse_matmul
// CHECK: %[[EMPTY_W:.*]] = tensor.empty() : tensor<3x4xf32>
// CHECK: %[[BROADCAST_W:.*]] = linalg.broadcast ins(%arg2 : tensor<4xf32>) outs(%[[EMPTY_W]] : tensor<3x4xf32>) dimensions = [0]
// CHECK: %[[SCALED_W:.*]] = arith.mulf %arg1, %[[BROADCAST_W]] : tensor<3x4xf32>
// CHECK: %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK: %[[BROADCAST_OUT:.*]] = linalg.broadcast ins(%arg3 : tensor<4xf32>) outs(%[[EMPTY_OUT]] : tensor<2x4xf32>) dimensions = [0]
// CHECK: %[[RESULT:.*]] = linalg.matmul ins(%arg0, %[[SCALED_W]] : tensor<2x3xf32>, tensor<3x4xf32>) outs(%[[BROADCAST_OUT]] : tensor<2x4xf32>)
// CHECK: return %[[RESULT]]
func.func @fuse_matmul(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> tensor<2x4xf32> {
  %0 = tensor.empty() : tensor<2x4xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
  %2 = tensor.empty() : tensor<2x4xf32>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<4xf32>) outs(%2 : tensor<2x4xf32>) dimensions = [0]
  %3 = arith.mulf %1, %broadcasted : tensor<2x4xf32>
  %4 = tensor.empty() : tensor<2x4xf32>
  %broadcasted_0 = linalg.broadcast ins(%arg3 : tensor<4xf32>) outs(%4 : tensor<2x4xf32>) dimensions = [0]
  %5 = arith.addf %3, %broadcasted_0 : tensor<2x4xf32>
  return %5 : tensor<2x4xf32>
}
