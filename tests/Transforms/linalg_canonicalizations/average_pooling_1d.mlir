// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x6x28xf32>)
  // CHECK: %[[out:.*]] = arith.constant dense<3.0
  // CHECK: %[[divided_cst:.*]] = arith.constant
  // CHECK-SAME: 5.000000e-01, 5.000000e-01
  // CHECK-SAME: tensor<6x6x2xf32>
  // CHECK: linalg.conv_1d_ncw_fcw
  // CHECK-SAME: strides = dense<2> : vector<1xi64>
  // CHECK-SAME: ins(%[[arg0]], %[[divided_cst]] : tensor<1x6x28xf32>, tensor<6x6x2xf32>) outs(%[[out]] : tensor<1x6x14xf32>)
  // CHECK: return
  func.func @main(%arg0: tensor<1x6x28xf32>) -> tensor<1x6x14xf32> {
    %cst_0 = arith.constant 3.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x6x14xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x6x14xf32>) -> tensor<1x6x14xf32>
    // filter constant doesn't affect output. Choosing 3 avoids other optimizations interfering
    %5 = arith.constant dense<6.0> : tensor<2xf32>
    %6 = linalg.pooling_ncw_sum {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%arg0, %5 : tensor<1x6x28xf32>, tensor<2xf32>) outs(%4 : tensor<1x6x14xf32>) -> tensor<1x6x14xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<1x6x14xf32>) outs(%3 : tensor<1x6x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.divf %in, %cst_1 : f32
      linalg.yield %32 : f32
    } -> tensor<1x6x14xf32>
    return %7 : tensor<1x6x14xf32>
  }
}
