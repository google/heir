// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// Rewrite a dilated conv2d kernel as a conv2d with a filter with 0s inserted
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x1x8x8xf32>)
  // CHECK-DAG: %[[out:.*]] = arith.constant {{.*}} tensor<1x1x4x4xf32>
  // CHECK-DAG: %[[new_filter:.*]] = arith.constant
  // CHECK-SAME: [1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 3.000000e+00]
  // CHECK-SAME: [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
  // CHECK-SAME: [4.000000e+00, 0.000000e+00, 5.000000e+00, 0.000000e+00, 6.000000e+00]
  // CHECK-SAME: [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
  // CHECK-SAME: [7.000000e+00, 0.000000e+00, 8.000000e+00, 0.000000e+00, 9.000000e+00]
  // CHECK-SAME: tensor<1x1x5x5xf32>
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-SAME: dilations = dense<1>
  // CHECK-SAME: strides = dense<1>
  // CHECK-SAME: ins(%[[arg0]], %[[new_filter]] : tensor<1x1x8x8xf32>, tensor<1x1x5x5xf32>) outs(%[[out]] : tensor<1x1x4x4xf32>)
  // CHECK: return
  func.func @main(%arg0: tensor<1x1x8x8xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4x4xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
    %filter = arith.constant dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00, 9.000000e+00]]]]> : tensor<1x1x3x3xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %filter : tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) outs(%1 : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>
  }
}

// -----

// Rewrite a conv2d kernel with different dilations per dimension (2 along H,
// 3 along W)
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x1x8x8xf32>)
  // CHECK-DAG: %[[out:.*]] = arith.constant {{.*}} tensor<1x1x6x2xf32>
  // CHECK-DAG: %[[new_filter:.*]] = arith.constant
  // CHECK-SAME: [1.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00, 3.000000e+00]
  // CHECK-SAME: [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
  // CHECK-SAME: [4.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e+00, 0.000000e+00, 0.000000e+00, 6.000000e+00]
  // CHECK-SAME: tensor<1x1x3x7xf32>
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-SAME: dilations = dense<1>
  // CHECK-SAME: strides = dense<1>
  // CHECK-SAME: ins(%[[arg0]], %[[new_filter]] : tensor<1x1x8x8xf32>, tensor<1x1x3x7xf32>) outs(%[[out]] : tensor<1x1x6x2xf32>)
  // CHECK: return
  func.func @main(%arg0: tensor<1x1x8x8xf32>) -> tensor<1x1x6x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x6x2xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x6x2xf32>) -> tensor<1x1x6x2xf32>
    %filter = arith.constant dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]]]> : tensor<1x1x2x3xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %filter : tensor<1x1x8x8xf32>, tensor<1x1x2x3xf32>) outs(%1 : tensor<1x1x6x2xf32>) -> tensor<1x1x6x2xf32>
    return %2 : tensor<1x1x6x2xf32>
  }
}
