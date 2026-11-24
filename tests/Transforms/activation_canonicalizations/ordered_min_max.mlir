// RUN: heir-opt --activation-canonicalizations --split-input-file %s | FileCheck %s

// torch's ELU and LeakyReLU split x into max(0, x) and min(0, x), using
// ordered predicates and a 0-d constant input broadcast into the body.

#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: @elu_split
// CHECK: linalg.generic
// CHECK: arith.maximumf %[[A:.*]], %[[B:.*]] : f32
// CHECK: linalg.generic
// CHECK: arith.minimumf %[[C:.*]], %[[D:.*]] : f32
// CHECK-NOT: arith.select
func.func @elu_split(%arg0: tensor<1x64xf32>) -> (tensor<1x64xf32>, tensor<1x64xf32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = tensor.empty() : tensor<1x64xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst, %arg0 : tensor<f32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %6 = arith.cmpf ogt, %in, %in_1 : f32
    %7 = arith.select %6, %in, %in_1 : f32
    linalg.yield %7 : f32
  } -> tensor<1x64xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst, %arg0 : tensor<f32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %6 = arith.cmpf olt, %in, %in_1 : f32
    %7 = arith.select %6, %in, %in_1 : f32
    linalg.yield %7 : f32
  } -> tensor<1x64xf32>
  return %1, %2 : tensor<1x64xf32>, tensor<1x64xf32>
}

// -----

// All ordered and unordered orderings are accepted.
// CHECK: @predicates
// CHECK-COUNT-2: arith.maximumf
// CHECK-COUNT-4: arith.minimumf
// CHECK-NOT: arith.select
func.func @predicates(%a: f32, %c: f32) -> (f32, f32, f32, f32, f32, f32) {
  %0 = arith.cmpf oge, %a, %c : f32
  %1 = arith.select %0, %a, %c : f32
  %2 = arith.cmpf uge, %a, %c : f32
  %3 = arith.select %2, %a, %c : f32
  %4 = arith.cmpf ole, %a, %c : f32
  %5 = arith.select %4, %a, %c : f32
  %6 = arith.cmpf ule, %a, %c : f32
  %7 = arith.select %6, %a, %c : f32
  %8 = arith.cmpf olt, %a, %c : f32
  %9 = arith.select %8, %a, %c : f32
  %10 = arith.cmpf ult, %a, %c : f32
  %11 = arith.select %10, %a, %c : f32
  return %1, %3, %5, %7, %9, %11 : f32, f32, f32, f32, f32, f32
}

// -----

// Equality comparisons and mismatched operands are left alone.
// CHECK: @not_min_max
// CHECK-COUNT-2: arith.select
// CHECK-NOT: arith.maximumf
// CHECK-NOT: arith.minimumf
func.func @not_min_max(%a: f32, %c: f32) -> (f32, f32) {
  %0 = arith.cmpf oeq, %a, %c : f32
  %1 = arith.select %0, %a, %c : f32
  %2 = arith.cmpf ogt, %a, %c : f32
  %3 = arith.select %2, %c, %a : f32
  return %1, %3 : f32, f32
}
