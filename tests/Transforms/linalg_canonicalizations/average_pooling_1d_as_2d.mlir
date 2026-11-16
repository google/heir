// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// torch-mlir lowers nn.AvgPool1d as a 2D pool over an expanded unit dimension,
// with strides [1, s]. The pool is rewritten as a conv_2d (folding in the
// average division), which is then downscaled to a conv_1d since its H window
// and output dimensions have size 1.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @avgpool1d
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x4x16xf32>)
  // CHECK-DAG: %[[kernel:.*]] = arith.constant dense<{{.*}}5.000000e-01, 5.000000e-01{{.*}}> : tensor<4x4x1x2xf32>
  // CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg0]]
  // CHECK-DAG: %[[input:.*]] = tensor.extract_slice %[[expanded]]{{.*}} to tensor<1x4x16xf32>
  // CHECK-DAG: %[[filter:.*]] = tensor.extract_slice %[[kernel]]{{.*}} to tensor<4x4x2xf32>
  // CHECK: linalg.conv_1d_ncw_fcw
  // CHECK-SAME: strides = dense<2> : tensor<1xi64>
  // CHECK-SAME: ins(%[[input]], %[[filter]] : tensor<1x4x16xf32>, tensor<4x4x2xf32>)
  // CHECK-SAME: -> tensor<1x4x8xf32>
  // CHECK-NOT: linalg.conv_2d_nchw_fchw
  // CHECK-NOT: linalg.pooling_nchw_sum
  // CHECK-NOT: arith.divf
  // CHECK: return
  func.func @avgpool1d(%arg0: tensor<1x4x16xf32>) -> tensor<1x4x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [1, 4, 1, 16] : tensor<1x4x16xf32> into tensor<1x4x1x16xf32>
    %0 = tensor.empty() : tensor<1x4x1x8xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    %2 = tensor.empty() : tensor<1x2xf32>
    %3 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<[1, 2]> : vector<2xi64>} ins(%expanded, %2 : tensor<1x4x1x16xf32>, tensor<1x2xf32>) outs(%1 : tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x4x1x8xf32>) outs(%0 : tensor<1x4x1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.divf %in, %cst_0 : f32
      linalg.yield %5 : f32
    } -> tensor<1x4x1x8xf32>
    %collapsed = tensor.collapse_shape %4 [[0], [1, 2], [3]] : tensor<1x4x1x8xf32> into tensor<1x4x8xf32>
    return %collapsed : tensor<1x4x8xf32>
  }
}

// -----

// A conv_2d without a size-1 window dimension is left alone.
// CHECK: func.func @no_unit_window
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: strides = dense<[1, 2]>
// CHECK-NOT: linalg.conv_1d_ncw_fcw
func.func @no_unit_window(%arg0: tensor<1x1x2x8xf32>, %filter: tensor<1x1x2x2xf32>) -> tensor<1x1x1x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1x1x4xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
  %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} ins(%arg0, %filter : tensor<1x1x2x8xf32>, tensor<1x1x2x2xf32>) outs(%1 : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
  return %2 : tensor<1x1x1x4xf32>
}
