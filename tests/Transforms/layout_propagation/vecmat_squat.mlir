// RUN: heir-opt --layout-propagation=ciphertext-size=8 --fold-convert-layout-into-assign-layout %s | FileCheck %s

// CHECK: func.func @vecmat_squat
func.func @vecmat_squat(%arg0: !secret.secret<tensor<5xf32>>) -> !secret.secret<tensor<3xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3xf32>
  // CHECK: %[[bias:.*]] = arith.constant dense<0.00
  // CHECK: %[[cst:.*]] = arith.constant
  // CHECK-SAME: tensor<5x3xf32>

  // Assign a layout to the matrix and bias
  // CHECK-DAG: tensor_ext.assign_layout %[[cst]]
  // CHECK-DAG: tensor_ext.assign_layout %[[bias]]
  %cst_0 = arith.constant dense<3.0> : tensor<5x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<5xf32>>) {
  ^body(%input0: tensor<5xf32>):
    // CHECK: linalg.vecmat
    // CHECK-SAME: secret.kernel = #kernel
    %1 = linalg.vecmat ins(%input0, %cst_0 : tensor<5xf32>, tensor<5x3xf32>) outs(%cst : tensor<3xf32>) -> tensor<3xf32>
    secret.yield %1 : tensor<3xf32>
    // CHECK: secret.yield
    // CHECK-NEXT: -> (!secret.secret<tensor<3xf32>>
  } -> !secret.secret<tensor<3xf32>>
  return %0 : !secret.secret<tensor<3xf32>>
}
