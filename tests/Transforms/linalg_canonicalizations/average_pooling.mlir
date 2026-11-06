// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x6x28x28xf32>)
  // CHECK-DAG: %[[out:.*]] = arith.constant dense<5.000000e-01>
  // CHECK-DAG: %[[divided_cst:.*]] = arith.constant dense<"0x0000803E0000803E0000803E0000803
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-SAME: strides = dense<2> : vector<2xi64>
  // CHECK-SAME: ins(%[[arg0]], %[[divided_cst]] : tensor<1x6x28x28xf32>, tensor<6x6x2x2xf32>) outs(%[[out]] : tensor<1x6x14x14xf32>)
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

// -----

// A 3x3 pool divided by 9 in a linalg.generic: the pattern folds the 1/9 into
// the kernel.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @divide_by_nine
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-NOT: arith.divf
  // CHECK-NOT: arith.mulf
  // CHECK: return
  func.func @divide_by_nine(%arg0: tensor<1x2x9x9xf32>) -> tensor<1x2x3x3xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %nine = arith.constant 9.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x2x3x3xf32>
    %4 = linalg.fill ins(%zero : f32) outs(%3 : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
    %5 = tensor.empty() : tensor<3x3xf32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<3> : vector<2xi64>} ins(%arg0, %5 : tensor<1x2x9x9xf32>, tensor<3x3xf32>) outs(%4 : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x2x3x3xf32>) outs(%3 : tensor<1x2x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.divf %in, %nine : f32
      linalg.yield %8 : f32
    } -> tensor<1x2x3x3xf32>
    return %7 : tensor<1x2x3x3xf32>
  }
}

// -----

// The same scaled by a multiplication with 1/4.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @multiply_by_quarter
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-NOT: arith.mulf
  // CHECK: return
  func.func @multiply_by_quarter(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %quarter = arith.constant 2.500000e-01 : f32
    %3 = tensor.empty() : tensor<1x2x2x2xf32>
    %4 = linalg.fill ins(%zero : f32) outs(%3 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %5 = tensor.empty() : tensor<2x2xf32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %5 : tensor<1x2x4x4xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x2x2x2xf32>) outs(%3 : tensor<1x2x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %quarter : f32
      linalg.yield %8 : f32
    } -> tensor<1x2x2x2xf32>
    return %7 : tensor<1x2x2x2xf32>
  }
}

// -----

// A constant divided by the pool is not an average; the division stays.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @constant_over_pool
  // CHECK: arith.divf
  func.func @constant_over_pool(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %four = arith.constant 4.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x2x2x2xf32>
    %4 = linalg.fill ins(%zero : f32) outs(%3 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %5 = tensor.empty() : tensor<2x2xf32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %5 : tensor<1x2x4x4xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x2x2x2xf32>) outs(%3 : tensor<1x2x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.divf %four, %in : f32
      linalg.yield %8 : f32
    } -> tensor<1x2x2x2xf32>
    return %7 : tensor<1x2x2x2xf32>
  }
}

// -----

// A generic that reads the pool transposed does not scale it elementwise, so
// it stays.

#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#tr = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
module {
  // CHECK: func.func @transposed_scale
  // CHECK: arith.mulf
  func.func @transposed_scale(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %quarter = arith.constant 2.500000e-01 : f32
    %3 = tensor.empty() : tensor<1x1x2x2xf32>
    %4 = linalg.fill ins(%zero : f32) outs(%3 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %5 = tensor.empty() : tensor<2x2xf32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %5 : tensor<1x1x4x4xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %7 = linalg.generic {indexing_maps = [#tr, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x1x2x2xf32>) outs(%3 : tensor<1x1x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %quarter : f32
      linalg.yield %8 : f32
    } -> tensor<1x1x2x2xf32>
    return %7 : tensor<1x1x2x2xf32>
  }
}

// -----

// The conv adds its init unscaled, so the pattern scales a constant init.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @scaled_init
  // CHECK: %[[init:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-SAME: outs(%[[init]]
  // CHECK-NOT: arith.mulf
  // CHECK: return
  func.func @scaled_init(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
    %four = arith.constant 4.000000e+00 : f32
    %quarter = arith.constant 2.500000e-01 : f32
    %3 = tensor.empty() : tensor<1x1x2x2xf32>
    %4 = linalg.fill ins(%four : f32) outs(%3 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %5 = tensor.empty() : tensor<2x2xf32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %5 : tensor<1x1x4x4xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x1x2x2xf32>) outs(%3 : tensor<1x1x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %quarter : f32
      linalg.yield %8 : f32
    } -> tensor<1x1x2x2xf32>
    return %7 : tensor<1x1x2x2xf32>
  }
}

// -----

// An integer sum pool becomes a conv with a filter of ones.

module {
  // CHECK: func.func @int_sum_pool
  // CHECK: linalg.conv_2d_nchw_fchw
  func.func @int_sum_pool(%arg0: tensor<1x1x4x4xi32>) -> tensor<1x1x2x2xi32> {
    %zero = arith.constant 0 : i32
    %3 = tensor.empty() : tensor<1x1x2x2xi32>
    %4 = linalg.fill ins(%zero : i32) outs(%3 : tensor<1x1x2x2xi32>) -> tensor<1x1x2x2xi32>
    %5 = tensor.empty() : tensor<2x2xi32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %5 : tensor<1x1x4x4xi32>, tensor<2x2xi32>) outs(%4 : tensor<1x1x2x2xi32>) -> tensor<1x1x2x2xi32>
    return %6 : tensor<1x1x2x2xi32>
  }
}
