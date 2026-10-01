// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x6x14x14xf32>)
  // CHECK: %[[cst:.*]] = arith.constant dense<4.000000e+00> : tensor<1x6x14x14xf32>
  // CHECK-NOT: linalg.generic
  // CHECK: arith.divf %[[arg0]], %[[cst]] : tensor<1x6x14x14xf32>
  // CHECK: return
  func.func @main(%arg0: tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32> {
    %cst_1 = arith.constant 4.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x6x14x14xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x6x14x14xf32>) outs(%3 : tensor<1x6x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.divf %in, %cst_1 : f32
      linalg.yield %32 : f32
    } -> tensor<1x6x14x14xf32>
    return %0 : tensor<1x6x14x14xf32>
  }
}

// -----


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @attrs
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x6x14x14xf32>)
  // CHECK: %[[cst:.*]] = arith.constant dense<4.000000e+00> : tensor<1x6x14x14xf32>
  // CHECK-NOT: linalg.generic
  // CHECK: arith.divf %[[arg0]], %[[cst]] {some_attr = "foo"} : tensor<1x6x14x14xf32>
  // CHECK: return
  func.func @attrs(%arg0: tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32> {
    %cst_1 = arith.constant 4.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x6x14x14xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], some_attr = "foo"} ins(%arg0 : tensor<1x6x14x14xf32>) outs(%3 : tensor<1x6x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.divf %in, %cst_1 : f32
      linalg.yield %32 : f32
    } -> tensor<1x6x14x14xf32>
    return %0 : tensor<1x6x14x14xf32>
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
  // CHECK: func.func @yield_external_constant
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1xi64>)
  // CHECK: %[[cst:.*]] = arith.constant dense<0> : tensor<1xi64>
  // CHECK: return %[[cst]]
  func.func @yield_external_constant(%arg0: tensor<1xi64>) -> tensor<1xi64> {
    %c0_i64 = arith.constant 0 : i64
    %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg0 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
  // CHECK: func.func @yield_external_arg
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1xi64>, %[[arg1:.*]]: i64)
  // CHECK: %[[splat:.*]] = tensor.splat %[[arg1]] : tensor<1xi64>
  // CHECK: return %[[splat]]
  func.func @yield_external_arg(%arg0: tensor<1xi64>, %arg1: i64) -> tensor<1xi64> {
    %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg0 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %arg1 : i64
    } -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }
}
