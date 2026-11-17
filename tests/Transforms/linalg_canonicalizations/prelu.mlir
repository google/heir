// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// torch's PReLU: select(x > 0, x, a * x), with the comparison, the scaling by
// the (broadcast) weight a, and the select in separate generics. It is
// rewritten as a * x + relu(x) - a * relu(x), so that it is approximated
// through ReLU rather than through the discontinuous step x > 0.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
// CHECK: @prelu
// CHECK-SAME: (%[[X:.*]]: tensor<1x64xf32>)
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<1x64xf32>
// CHECK-DAG: %[[A:.*]] = arith.constant dense<2.500000e-01> : tensor<1x1xf32>
// CHECK: %[[AX:.*]] = linalg.generic {{.*}} ins(%[[A]], %[[X]] :
// CHECK: %[[RELU:.*]] = arith.maximumf %[[X]], %[[ZERO]] : tensor<1x64xf32>
// CHECK: %[[ARELU:.*]] = linalg.generic {{.*}} ins(%[[A]], %[[RELU]] :
// CHECK: %[[DIFF:.*]] = arith.subf %[[RELU]], %[[ARELU]] : tensor<1x64xf32>
// CHECK: %[[RES:.*]] = arith.addf %[[AX]], %[[DIFF]] : tensor<1x64xf32>
// CHECK-NOT: arith.select
// CHECK-NOT: arith.cmpf
// CHECK: return %[[RES]]
func.func @prelu(%arg0: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %cst = arith.constant dense<2.500000e-01> : tensor<1x1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x64xi1>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x64xf32>) outs(%0 : tensor<1x64xi1>) {
  ^bb0(%in: f32, %out: i1):
    %5 = arith.cmpf ogt, %in, %cst_0 : f32
    linalg.yield %5 : i1
  } -> tensor<1x64xi1>
  %2 = tensor.empty() : tensor<1x64xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst, %arg0 : tensor<1x1xf32>, tensor<1x64xf32>) outs(%2 : tensor<1x64xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %5 = arith.mulf %in, %in_1 : f32
    linalg.yield %5 : f32
  } -> tensor<1x64xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %arg0, %3 : tensor<1x64xi1>, tensor<1x64xf32>, tensor<1x64xf32>) outs(%2 : tensor<1x64xf32>) {
  ^bb0(%in: i1, %in_1: f32, %in_2: f32, %out: f32):
    %5 = arith.select %in, %in_1, %in_2 : f32
    linalg.yield %5 : f32
  } -> tensor<1x64xf32>
  return %4 : tensor<1x64xf32>
}

// -----

// The same with the scaling as an elementwise mulf and the comparison written
// as 0 < x.
// CHECK: @prelu_mulf
// CHECK-SAME: (%[[X:.*]]: tensor<4xf32>)
// CHECK: %[[RELU:.*]] = arith.maximumf %[[X]]
// CHECK: arith.subf %[[RELU]]
// CHECK: arith.addf
// CHECK-NOT: arith.select
func.func @prelu_mulf(%x: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %a = arith.constant dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf32>
  %c = arith.cmpf olt, %zero, %x : tensor<4xf32>
  %ax = arith.mulf %a, %x : tensor<4xf32>
  %0 = arith.select %c, %x, %ax : tensor<4xi1>, tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A non-constant slope is not a PReLU.
// CHECK: @not_prelu
// CHECK: arith.select
// CHECK-NOT: arith.maximumf
func.func @not_prelu(%x: tensor<4xf32>, %a: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %c = arith.cmpf ogt, %x, %zero : tensor<4xf32>
  %ax = arith.mulf %a, %x : tensor<4xf32>
  %0 = arith.select %c, %x, %ax : tensor<4xi1>, tensor<4xf32>
  return %0 : tensor<4xf32>
}
