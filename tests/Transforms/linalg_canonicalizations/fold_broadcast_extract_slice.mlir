// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// CHECK: func @test_dense_resource
// CHECK: %[[CST:.*]] = arith.constant dense<[3.300000e+00, 4.400000e+00]> : tensor<2xf32>
// CHECK: return %[[CST]] : tensor<2xf32>
func.func @test_dense_resource() -> tensor<2xf32> {
  %cst = arith.constant dense_resource<resource1> : tensor<4xf32>
  %0 = tensor.empty() : tensor<4x2xf32>
  %1 = linalg.broadcast ins(%cst : tensor<4xf32>) outs(%0 : tensor<4x2xf32>) dimensions = [1]
  %2 = tensor.extract_slice %1[2, 0] [2, 1] [1, 1] : tensor<4x2xf32> to tensor<2x1xf32>
  %3 = tensor.collapse_shape %2 [[0, 1]] : tensor<2x1xf32> into tensor<2xf32>
  return %3 : tensor<2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      resource1: "0x04000000CDCC8C3FCDCC0C4033335340CDCC8C40"
    }
  }
#-}

// -----

// CHECK: func @test_inline_constant
// CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<1xf32>
// CHECK: return %[[CST]] : tensor<1xf32>
func.func @test_inline_constant() -> tensor<1xf32> {
  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %0 = tensor.empty() : tensor<4x2xf32>
  %1 = linalg.broadcast ins(%cst : tensor<4xf32>) outs(%0 : tensor<4x2xf32>) dimensions = [1]
  %2 = tensor.extract_slice %1[1, 1] [1, 1] [1, 1] : tensor<4x2xf32> to tensor<1x1xf32>
  %3 = tensor.collapse_shape %2 [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
  return %3 : tensor<1xf32>
}

// -----

// CHECK: func @test_no_match_arg
// CHECK: linalg.broadcast
// CHECK: tensor.extract_slice
// CHECK: return
func.func @test_no_match_arg(%arg0: tensor<4xf32>) -> tensor<1xf32> {
  %0 = tensor.empty() : tensor<4x2xf32>
  %1 = linalg.broadcast ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4x2xf32>) dimensions = [1]
  %2 = tensor.extract_slice %1[1, 1] [1, 1] [1, 1] : tensor<4x2xf32> to tensor<1x1xf32>
  %3 = tensor.collapse_shape %2 [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
  return %3 : tensor<1xf32>
}

// -----

// CHECK: func @test_fold_broadcast_dense_resource
// CHECK: %[[CST:.*]] = arith.constant dense<{{.*}}>
// CHECK: return %[[CST]] : tensor<4x2xf32>
func.func @test_fold_broadcast_dense_resource() -> tensor<4x2xf32> {
  %cst = arith.constant dense_resource<resource1> : tensor<4xf32>
  %0 = tensor.empty() : tensor<4x2xf32>
  %1 = linalg.broadcast ins(%cst : tensor<4xf32>) outs(%0 : tensor<4x2xf32>) dimensions = [1]
  return %1 : tensor<4x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      resource1: "0x04000000CDCC8C3FCDCC0C4033335340CDCC8C40"
    }
  }
#-}

// -----

// CHECK: func @test_fold_broadcast_splat_dense_resource
// CHECK: %[[CST:.*]] = arith.constant dense<5.5
// CHECK: return %[[CST]] : tensor<4x2xf32>
func.func @test_fold_broadcast_splat_dense_resource() -> tensor<4x2xf32> {
  %cst = arith.constant dense_resource<resource2> : tensor<4xf32>
  %0 = tensor.empty() : tensor<4x2xf32>
  %1 = linalg.broadcast ins(%cst : tensor<4xf32>) outs(%0 : tensor<4x2xf32>) dimensions = [1]
  return %1 : tensor<4x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      resource2: "0x040000003333B0403333B0403333B0403333B040"
    }
  }
#-}
