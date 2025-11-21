// RUN: heir-opt --activation-canonicalizations --canonicalize --split-input-file %s | FileCheck %s

// CHECK: func.func @relu_signed
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> i32
// CHECK: %[[A:.*]] = arith.constant 0
// CHECK-NEXT: arith.maxsi %[[ARG0]], %[[A]]
// CHECK: return
func.func @relu_signed(%arg0: i32) -> i32 {
  %cst_0 = arith.constant 0 : i32
  %0 = arith.cmpi sgt, %arg0, %cst_0 : i32
  %1 = arith.select %0, %arg0, %cst_0 : i32
  return %1 : i32
}

// -----

// CHECK: func.func @relu_unsigned
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> i32
// CHECK: %[[A:.*]] = arith.constant 1
// CHECK-NEXT: arith.maxui %[[ARG0]], %[[A]]
// CHECK: return
func.func @relu_unsigned(%arg0: i32) -> i32 {
  %cst_1 = arith.constant 1 : i32
  %0 = arith.cmpi ugt, %arg0, %cst_1 : i32
  %1 = arith.select %0, %arg0, %cst_1 : i32
  return %1 : i32
}

// -----

// CHECK: func.func @sigmoid
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> f32
// CHECK: math_ext.sigmoid %[[ARG0]]
// CHECK: return
func.func @sigmoid(%arg0: f32) -> f32 {
  %cst_one = arith.constant 1.0 : f32
  %neg_x = arith.negf %arg0 : f32
  %exp_neg_x = math.exp %neg_x : f32
  %one_plus_exp = arith.addf %exp_neg_x, %cst_one : f32
  %result = arith.divf %cst_one, %one_plus_exp : f32
  return %result : f32
}

// -----

// CHECK: func.func @sigmoid_tensor
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xf32>) -> tensor<1xf32>
// CHECK: math_ext.sigmoid %[[ARG0]]
// CHECK: return
func.func @sigmoid_tensor(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %cst_one = arith.constant dense<1.0> : tensor<1xf32>
  %neg_x = arith.negf %arg0 : tensor<1xf32>
  %exp_neg_x = math.exp %neg_x : tensor<1xf32>
  %one_plus_exp = arith.addf %exp_neg_x, %cst_one : tensor<1xf32>
  %result = arith.divf %cst_one, %one_plus_exp : tensor<1xf32>
  return %result : tensor<1xf32>
}
