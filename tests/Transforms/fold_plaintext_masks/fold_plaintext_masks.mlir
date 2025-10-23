// RUN: heir-opt %s --fold-plaintext-masks | FileCheck %s

// CHECK: @repeated_mask
// CHECK-COUNT-1: arith.muli
func.func @repeated_mask(%arg0: tensor<32xi32>) -> tensor<32xi32> {
  %cst_1 = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<32xi32>
  %2 = arith.muli %arg0, %cst_1 : tensor<32xi32>
  %3 = arith.muli %2, %cst_1 : tensor<32xi32>
  %4 = arith.muli %3, %cst_1 : tensor<32xi32>
  %5 = arith.muli %4, %cst_1 : tensor<32xi32>
  %6 = arith.muli %5, %cst_1 : tensor<32xi32>
  return %6 : tensor<32xi32>
}

// CHECK: @intersected_mask
// CHECK-SAME: ([[arg0:%.*]]: tensor<6xi32>) -> tensor<6xi32>
// CHECK-NEXT: [[mask:%.*]] = arith.constant dense<[0, 1, 1, 0, 0, 0]> : tensor<6xi32>
// CHECK-NEXT: arith.muli [[arg0]], [[mask]]
// CHECK: return
func.func @intersected_mask(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  %cst_1 = arith.constant dense<[1, 1, 1, 0, 0, 0]> : tensor<6xi32>
  %cst_2 = arith.constant dense<[0, 1, 1, 1, 1, 1]> : tensor<6xi32>
  %1 = arith.muli %arg0, %cst_1 : tensor<6xi32>
  %2 = arith.muli %1, %cst_2 : tensor<6xi32>
  return %2 : tensor<6xi32>
}

// CHECK: @intersected_mask_float
// CHECK-SAME: ([[arg0:%.*]]: tensor<6xf32>) -> tensor<6xf32>
// CHECK-NEXT: [[mask:%.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<6xf32>
// CHECK-NEXT: arith.mulf [[arg0]], [[mask]]
// CHECK: return
func.func @intersected_mask_float(%arg0: tensor<6xf32>) -> tensor<6xf32> {
  %cst_1 = arith.constant dense<[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]> : tensor<6xf32>
  %cst_2 = arith.constant dense<[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]> : tensor<6xf32>
  %1 = arith.mulf %arg0, %cst_1 : tensor<6xf32>
  %2 = arith.mulf %1, %cst_2 : tensor<6xf32>
  return %2 : tensor<6xf32>
}
