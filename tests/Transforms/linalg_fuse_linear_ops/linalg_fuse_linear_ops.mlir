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

// -----

// CHECK: func.func @fuse_matvec
// CHECK: %[[EMPTY_W:.*]] = tensor.empty() : tensor<2x3xf32>
// CHECK: %[[BROADCAST_W:.*]] = linalg.broadcast ins(%arg2 : tensor<2xf32>) outs(%[[EMPTY_W]] : tensor<2x3xf32>) dimensions = [1]
// CHECK: %[[SCALED_W:.*]] = arith.mulf %arg0, %[[BROADCAST_W]] : tensor<2x3xf32>
// CHECK: %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<2xf32>
// CHECK: %[[BROADCAST_OUT:.*]] = linalg.broadcast ins(%arg3 : tensor<2xf32>) outs(%[[EMPTY_OUT]] : tensor<2xf32>) dimensions = []
// CHECK: %[[RESULT:.*]] = linalg.matvec ins(%[[SCALED_W]], %arg1 : tensor<2x3xf32>, tensor<3xf32>) outs(%[[BROADCAST_OUT]] : tensor<2xf32>)
// CHECK: return %[[RESULT]]
func.func @fuse_matvec(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xf32> {
  %0 = tensor.empty() : tensor<2xf32>
  %1 = linalg.matvec ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3xf32>) outs(%0 : tensor<2xf32>) -> tensor<2xf32>
  %2 = arith.mulf %1, %arg2 : tensor<2xf32>
  %3 = arith.addf %2, %arg3 : tensor<2xf32>
  return %3 : tensor<2xf32>
}

// -----

// CHECK: func.func @fuse_vecmat
// CHECK: %[[EMPTY_W:.*]] = tensor.empty() : tensor<3x4xf32>
// CHECK: %[[BROADCAST_W:.*]] = linalg.broadcast ins(%arg2 : tensor<4xf32>) outs(%[[EMPTY_W]] : tensor<3x4xf32>) dimensions = [0]
// CHECK: %[[SCALED_W:.*]] = arith.mulf %arg1, %[[BROADCAST_W]] : tensor<3x4xf32>
// CHECK: %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[BROADCAST_OUT:.*]] = linalg.broadcast ins(%arg3 : tensor<4xf32>) outs(%[[EMPTY_OUT]] : tensor<4xf32>) dimensions = []
// CHECK: %[[RESULT:.*]] = linalg.vecmat ins(%arg0, %[[SCALED_W]] : tensor<3xf32>, tensor<3x4xf32>) outs(%[[BROADCAST_OUT]] : tensor<4xf32>)
// CHECK: return %[[RESULT]]
func.func @fuse_vecmat(%arg0: tensor<3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = linalg.vecmat ins(%arg0, %arg1 : tensor<3xf32>, tensor<3x4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  %2 = arith.mulf %1, %arg2 : tensor<4xf32>
  %3 = arith.addf %2, %arg3 : tensor<4xf32>
  return %3 : tensor<4xf32>
}

// -----

// CHECK: func.func @fuse_conv_2d
// CHECK: %[[EMPTY_W:.*]] = tensor.empty() : tensor<8x3x3x3xf32>
// CHECK: %[[BROADCAST_W:.*]] = linalg.broadcast ins(%arg2 : tensor<8xf32>) outs(%[[EMPTY_W]] : tensor<8x3x3x3xf32>) dimensions = [1, 2, 3]
// CHECK: %[[SCALED_W:.*]] = arith.mulf %arg1, %[[BROADCAST_W]] : tensor<8x3x3x3xf32>
// CHECK: %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<1x8x14x14xf32>
// CHECK: %[[BROADCAST_OUT:.*]] = linalg.broadcast ins(%arg3 : tensor<8xf32>) outs(%[[EMPTY_OUT]] : tensor<1x8x14x14xf32>) dimensions = [0, 2, 3]
// CHECK: %[[RESULT:.*]] = linalg.conv_2d_nchw_fchw ins(%arg0, %[[SCALED_W]] : tensor<1x3x16x16xf32>, tensor<8x3x3x3xf32>) outs(%[[BROADCAST_OUT]] : tensor<1x8x14x14xf32>)
// CHECK: return %[[RESULT]]
func.func @fuse_conv_2d(%arg0: tensor<1x3x16x16xf32>, %arg1: tensor<8x3x3x3xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> tensor<1x8x14x14xf32> {
  %0 = tensor.empty() : tensor<1x8x14x14xf32>
  %1 = linalg.conv_2d_nchw_fchw ins(%arg0, %arg1 : tensor<1x3x16x16xf32>, tensor<8x3x3x3xf32>) outs(%0 : tensor<1x8x14x14xf32>) -> tensor<1x8x14x14xf32>
  %2 = tensor.empty() : tensor<1x8x14x14xf32>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<8xf32>) outs(%2 : tensor<1x8x14x14xf32>) dimensions = [0, 2, 3]
  %3 = arith.mulf %1, %broadcasted : tensor<1x8x14x14xf32>
  %4 = tensor.empty() : tensor<1x8x14x14xf32>
  %broadcasted_0 = linalg.broadcast ins(%arg3 : tensor<8xf32>) outs(%4 : tensor<1x8x14x14xf32>) dimensions = [0, 2, 3]
  %5 = arith.addf %3, %broadcasted_0 : tensor<1x8x14x14xf32>
  return %5 : tensor<1x8x14x14xf32>
}

// -----

// CHECK: func.func @fuse_matmul_with_bias
// CHECK: %[[EMPTY_W:.*]] = tensor.empty() : tensor<3x4xf32>
// CHECK: %[[BROADCAST_W:.*]] = linalg.broadcast ins(%arg2 : tensor<4xf32>) outs(%[[EMPTY_W]] : tensor<3x4xf32>) dimensions = [0]
// CHECK: %[[SCALED_W:.*]] = arith.mulf %arg1, %[[BROADCAST_W]] : tensor<3x4xf32>
// CHECK: %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK: %[[BROADCAST_OUT:.*]] = linalg.broadcast ins(%arg3 : tensor<4xf32>) outs(%[[EMPTY_OUT]] : tensor<2x4xf32>) dimensions = [0]
// CHECK: %[[NEW_OUTS:.*]] = arith.addf %arg4, %[[BROADCAST_OUT]] : tensor<2x4xf32>
// CHECK: %[[RESULT:.*]] = linalg.matmul ins(%arg0, %[[SCALED_W]] : tensor<2x3xf32>, tensor<3x4xf32>) outs(%[[NEW_OUTS]] : tensor<2x4xf32>)
// CHECK: return %[[RESULT]]
func.func @fuse_matmul_with_bias(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>, %arg4: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%arg4 : tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = tensor.empty() : tensor<2x4xf32>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<4xf32>) outs(%1 : tensor<2x4xf32>) dimensions = [0]
  %2 = arith.mulf %0, %broadcasted : tensor<2x4xf32>
  %3 = tensor.empty() : tensor<2x4xf32>
  %broadcasted_0 = linalg.broadcast ins(%arg3 : tensor<4xf32>) outs(%3 : tensor<2x4xf32>) dimensions = [0]
  %4 = arith.addf %2, %broadcasted_0 : tensor<2x4xf32>
  return %4 : tensor<2x4xf32>
}
