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

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: func.func @sigmoid_generic
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x16x10x10xf32>) -> tensor<1x16x10x10xf32>
// CHECK: linalg.generic
// CHECK-SAME: domain_lower = -1.200000e+01
// CHECK-SAME: domain_upper = 1.200000e+01
// CHECK: math_ext.sigmoid
// CHECK: linalg.yield
// CHECK: return
func.func @sigmoid_generic(%arg0: tensor<1x16x10x10xf32>) -> tensor<1x16x10x10xf32> {
  %0 = tensor.empty() : tensor<1x16x10x10xf32>
  %cst_1 = arith.constant 1.0 : f32
  %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], domain_lower = -12.0, domain_upper = 12.0 } ins(%arg0 : tensor<1x16x10x10xf32>) outs(%0 : tensor<1x16x10x10xf32>) {
  ^bb0(%in: f32, %out: f32):
    %32 = arith.negf %in : f32
    %33 = math.exp %32 : f32
    %34 = arith.addf %33, %cst_1 : f32
    %35 = arith.divf %cst_1, %34 : f32
    linalg.yield %35 : f32
  } -> tensor<1x16x10x10xf32>
  return %10 : tensor<1x16x10x10xf32>
}
