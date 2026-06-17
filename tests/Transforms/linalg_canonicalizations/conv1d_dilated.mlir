// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// Rewrite a dilated conv1d kernel as a conv1d with a filter with 0s inserted
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x1x8xf32>)
  // CHECK-DAG: %[[out:.*]] = arith.constant {{.*}} tensor<1x1x4xf32>
  // CHECK-DAG: %[[new_filter:.*]] = arith.constant
  // CHECK-SAME: 1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 3.000000e+00
  // CHECK-SAME: tensor<1x1x5xf32>
  // CHECK: linalg.conv_1d_ncw_fcw
  // CHECK-SAME: dilations = dense<1>
  // CHECK-SAME: strides = dense<1>
  // CHECK-SAME: ins(%[[arg0]], %[[new_filter]] : tensor<1x1x8xf32>, tensor<1x1x5xf32>) outs(%[[out]] : tensor<1x1x4xf32>)
  // CHECK: return
  func.func @main(%arg0: tensor<1x1x8xf32>) -> tensor<1x1x4xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    %filter = arith.constant dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x3xf32>
    %2 = linalg.conv_1d_ncw_fcw {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %filter : tensor<1x1x8xf32>, tensor<1x1x3xf32>) outs(%1 : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    return %2 : tensor<1x1x4xf32>
  }
}

// -----

// Rewrite a dilation=3 conv1d kernel as a conv1d with a filter with 0s inserted
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x1x8xf32>)
  // CHECK-DAG: %[[out:.*]] = arith.constant {{.*}} tensor<1x1x2xf32>
  // CHECK-DAG: %[[new_filter:.*]] = arith.constant
  // CHECK-SAME: 1.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00, 3.000000e+00
  // CHECK-SAME: tensor<1x1x7xf32>
  // CHECK: linalg.conv_1d_ncw_fcw
  // CHECK-SAME: dilations = dense<1>
  // CHECK-SAME: strides = dense<1>
  // CHECK-SAME: ins(%[[arg0]], %[[new_filter]] : tensor<1x1x8xf32>, tensor<1x1x7xf32>) outs(%[[out]] : tensor<1x1x2xf32>)
  // CHECK: return
  func.func @main(%arg0: tensor<1x1x8xf32>) -> tensor<1x1x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x2xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    %filter = arith.constant dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x3xf32>
    %2 = linalg.conv_1d_ncw_fcw {dilations = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %filter : tensor<1x1x8xf32>, tensor<1x1x3xf32>) outs(%1 : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    return %2 : tensor<1x1x2xf32>
  }
}

// -----

module {
  // CHECK: func.func @main_dense_resource
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x1x8xf32>)
  // CHECK-DAG: %[[out:.*]] = arith.constant {{.*}} tensor<1x1x4xf32>
  // CHECK-DAG: %[[new_filter:.*]] = arith.constant
  // CHECK-SAME: 1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 3.000000e+00
  // CHECK-SAME: tensor<1x1x5xf32>
  // CHECK: linalg.conv_1d_ncw_fcw
  // CHECK-SAME: dilations = dense<1>
  // CHECK-SAME: strides = dense<1>
  // CHECK-SAME: ins(%[[arg0]], %[[new_filter]] : tensor<1x1x8xf32>, tensor<1x1x5xf32>) outs(%[[out]] : tensor<1x1x4xf32>)
  // CHECK: return
  func.func @main_dense_resource(%arg0: tensor<1x1x8xf32>) -> tensor<1x1x4xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    %filter = arith.constant dense_resource<filter_data> : tensor<1x1x3xf32>
    %2 = linalg.conv_1d_ncw_fcw {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %filter : tensor<1x1x8xf32>, tensor<1x1x3xf32>) outs(%1 : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    return %2 : tensor<1x1x4xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      filter_data: "0x040000000000803F0000004000004040"
    }
  }
#-}
