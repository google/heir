// RUN: heir-opt --apply-folders --split-input-file %s | FileCheck %s

// a + 0.0 = a

// CHECK: func.func @add_zero
// CHECK-SAME: (%[[arg:.*]]: tensor<2x2xf32>)
// CHECK: return %[[arg]]
func.func @add_zero(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = arith.constant dense<0.0> : tensor<2x2xf32>
  %1 = arith.addf %arg0, %0 : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// -----

// add(empty, a) = a

// CHECK: func.func @add_empty
// CHECK-SAME: (%[[arg:.*]]: tensor<2x2xf32>)
// CHECK: return %[[arg]]
func.func @add_empty(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = arith.addf %arg0, %0 : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
