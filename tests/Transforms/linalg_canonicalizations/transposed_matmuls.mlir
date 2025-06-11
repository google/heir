// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<512x784xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<784xf32>)
  // CHECK: %[[cst:.*]] = arith.constant dense<0.{{0*}}e+00> : tensor<512xf32>
  // CHECK: %[[v0:.*]] = linalg.matvec ins(%[[arg0]], %[[arg1]] : tensor<512x784xf32>, tensor<784xf32>) outs(%[[cst]] : tensor<512xf32>)
  // CHECK: return %[[v0]] : tensor<512xf32>
  func.func @main(%arg0: tensor<512x784xf32>, %arg2: tensor<784xf32>) -> tensor<512xf32> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = tensor.empty() : tensor<784x512xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<512x784xf32>) outs(%0 : tensor<784x512xf32>) permutation = [1, 0]
    %1 = linalg.vecmat ins(%arg2, %transposed : tensor<784xf32>, tensor<784x512xf32>) outs(%cst_0 : tensor<512xf32>) -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
}

// -----

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<784x512xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<784xf32>)
  // CHECK: %[[cst:.*]] = arith.constant dense<0.{{0*}}e+00> : tensor<512xf32>
  // CHECK: %[[v0:.*]] = linalg.vecmat ins(%[[arg1]], %[[arg0]] : tensor<784xf32>, tensor<784x512xf32>) outs(%[[cst]] : tensor<512xf32>)
  // CHECK: return %[[v0]] : tensor<512xf32>
  func.func @main(%arg0: tensor<784x512xf32>, %arg2: tensor<784xf32>) -> tensor<512xf32> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = tensor.empty() : tensor<512x784xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<784x512xf32>) outs(%0 : tensor<512x784xf32>) permutation = [1, 0]
    %1 = linalg.matvec ins(%transposed, %arg2 : tensor<512x784xf32>, tensor<784xf32>) outs(%cst_0 : tensor<512xf32>) -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
}
