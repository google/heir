// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x6x28x28xf32>)
  // CHECK-DAG: %[[divided_cst:.*]] = arith.constant dense<2.500000e-01>
  // CHECK-DAG: %[[out:.*]] = tensor.empty() : tensor<14x14xf32>
  // CHECK-DAG: %[[extracted:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK: linalg.conv_2d ins(%[[extracted]], %[[divided_cst]] : tensor<28x28xf32>, tensor<2x2xf32>) outs(%[[out]] : tensor<14x14xf32>)
  // CHECK-COUNT-5: linalg.conv_2d
  // CHECK: return
  func.func @main(%arg0: tensor<1x6x28x28xf32>) -> tensor<1x6x14x14xf32> {
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x6x14x14xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32>
    // filter constant doesn't affect output
    %5 = arith.constant dense<6.0> : tensor<2x2xf32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %5 : tensor<1x6x28x28xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x6x14x14xf32>) outs(%3 : tensor<1x6x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.divf %in, %cst_1 : f32
      linalg.yield %32 : f32
    } -> tensor<1x6x14x14xf32>
    return %7 : tensor<1x6x14x14xf32>
  }
}
