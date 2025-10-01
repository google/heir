// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  // CHECK: func.func @main
  // CHECK-NOT: linalg.generic
  // CHECK: arith.cmpf
  // CHECK-SAME: tensor<1x6x28x28xf32>
  // CHECK: arith.select
  // CHECK-SAME: tensor<1x6x28x28xf32>
  // CHECK: return
  func.func @main(%1: tensor<1x6x28x28xf32> {secret.secret}) -> tensor<1x6x28x28xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x6x28x28xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x6x28x28xf32>) outs(%0 : tensor<1x6x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.cmpf ugt, %in, %cst_0 : f32
      %33 = arith.select %32, %in, %cst_0 : f32
      linalg.yield %33 : f32
    } -> tensor<1x6x28x28xf32>
    return %2 : tensor<1x6x28x28xf32>
  }
}
