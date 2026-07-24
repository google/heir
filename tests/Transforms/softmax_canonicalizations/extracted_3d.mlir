// RUN: heir-opt --softmax-canonicalizations --canonicalize %s | FileCheck %s

// CHECK: func.func @main(%[[ARG:.*]]: tensor<4x16x64xf32> {secret.secret})
// CHECK-NEXT: %[[RES:.*]] = math_ext.softmax %[[ARG]] {dimension = 2 : i64} : tensor<4x16x64xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4x16x64xf32>

// -----// IR Dump After ReductionCanonicalizations: reduction-canonicalizations //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4x16x64xf32> {secret.secret}) -> tensor<4x16x64xf32> {
    %cst = arith.constant dense<0xFF800000> : tensor<4x16xf32>
    %cst_0 = arith.constant dense<0> : tensor<4x16xi64>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x16xf32>
    %0:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<4x16x64xf32>) outs(%cst, %cst_0 : tensor<4x16xf32>, tensor<4x16xi64>) {
    ^bb0(%in: f32, %out: f32, %out_3: i64):
      %6 = linalg.index 2 : index
      %7 = arith.index_cast %6 : index to i64
      %8 = arith.maximumf %in, %out : f32
      %9 = arith.cmpf ogt, %in, %out : f32
      %10 = arith.select %9, %7, %out_3 : i64
      linalg.yield %8, %10 : f32, i64
    } -> (tensor<4x16xf32>, tensor<4x16xi64>)
    %1 = tensor.empty() : tensor<4x16x64xf32>
    %broadcasted = linalg.broadcast ins(%0#0 : tensor<4x16xf32>) outs(%1 : tensor<4x16x64xf32>) dimensions = [2]
    %2 = arith.subf %arg0, %broadcasted : tensor<4x16x64xf32>
    %3 = math.exp %2 : tensor<4x16x64xf32>
    %reduced = linalg.reduce ins(%3 : tensor<4x16x64xf32>) outs(%cst_1 : tensor<4x16xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %6 = arith.addf %in, %init : f32
        linalg.yield %6 : f32
      }
    %4 = tensor.empty() : tensor<4x16x64xf32>
    %broadcasted_2 = linalg.broadcast ins(%reduced : tensor<4x16xf32>) outs(%4 : tensor<4x16x64xf32>) dimensions = [2]
    %5 = arith.divf %3, %broadcasted_2 : tensor<4x16x64xf32>
    return %5 : tensor<4x16x64xf32>
  }
}
