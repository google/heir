// RUN: heir-opt %s --drop-unit-dims --canonicalize --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<10xf32>)
  func.func @main(%arg0: tensor<10xf32>) -> tensor<1x10xf32> {
    // CHECK: %[[cst:.*]] = arith.constant dense<{{.*}}> : tensor<10xf32>
    %cst = arith.constant dense<2.0> : tensor<10xf32>
    %0 = tensor.empty() : tensor<1x10xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 10] : tensor<10xf32> into tensor<1x10xf32>
    // CHECK: %[[v2:.*]] = linalg.generic
    // CHECK-SAME: ins(%[[arg0]], %[[cst]] : tensor<10xf32>, tensor<10xf32>)
    // CHECK: -> tensor<10xf32>
    // CHECK: %[[inserted:.*]] = tensor.expand_shape %[[v2]]
    // CHECK: return %[[inserted]]
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded, %cst : tensor<1x10xf32>, tensor<10xf32>) outs(%0 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<1x10xf32>
    return %1 : tensor<1x10xf32>
  }
}
