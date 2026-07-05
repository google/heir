// RUN: heir-opt --broadcast-canonicalizations --split-input-file %s | FileCheck %s
// This tests a broadcast created from a torch-mlir layernorm.

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<2x64x768xf32> {secret.secret}) -> tensor<2x64x768xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x64xf32>
    %cst_2 = arith.constant dense<7.680000e-02> : tensor<2x64xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x64x768xf32>) outs(%cst : tensor<2x64xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %0 = arith.mulf %reduced, %cst_2 : tensor<2x64xf32>
    %1 = tensor.empty() : tensor<2x64x768xf32>
    // CHECK-NOT: linalg.broadcast ins(%0 : tensor<2x64xf32>) outs(%1 : tensor<2x64x768xf32>) dimensions = [2]
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<2x64xf32>) outs(%1 : tensor<2x64x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<2x64x768xf32>
    %3 = arith.subf %arg0, %2 : tensor<2x64x768xf32>
    %4 = arith.mulf %3, %3 : tensor<2x64x768xf32>
    return %4 : tensor<2x64x768xf32>
  }
}
