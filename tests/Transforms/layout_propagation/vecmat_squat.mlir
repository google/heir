// RUN: heir-opt --layout-propagation=min-slot-count=8 --fold-convert-layout-into-assign-layout %s | FileCheck %s

// CHECK: func.func @vecmat_squat
func.func @vecmat_squat(%arg0: !secret.secret<tensor<5xf32>>) -> !secret.secret<tensor<3xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3xf32>
  // CHECK: %[[bias:.*]] = arith.constant dense<0.00
  // CHECK: %[[cst:.*]] = arith.constant
  // CHECK-SAME: tensor<5x3xf32>

  // No vecmat kernel: the op is generalized and the cleartext matrix gets no
  // layout.
  // CHECK-NOT: tensor_ext.assign_layout %[[cst]]
  // CHECK: tensor_ext.assign_layout %[[bias]]
  %cst_0 = arith.constant dense<3.0> : tensor<5x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<5xf32>>) {
  ^body(%input0: tensor<5xf32>):
    // CHECK: linalg.generic
    // CHECK-SAME: tensor_ext.layout
    %1 = linalg.vecmat ins(%input0, %cst_0 : tensor<5xf32>, tensor<5x3xf32>) outs(%cst : tensor<3xf32>) -> tensor<3xf32>
    secret.yield %1 : tensor<3xf32>
    // CHECK: secret.yield
    // CHECK-NEXT: -> (!secret.secret<tensor<3xf32>>
  } -> !secret.secret<tensor<3xf32>>
  return %0 : !secret.secret<tensor<3xf32>>
}
