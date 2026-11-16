// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// torch-mlir lowers nn.MaxPool1d as a 2D max pool over an expanded unit
// dimension, with strides [1, s]. The pool is downscaled to a 1D pool, which is
// then rewritten as the elementwise max of one selection convolution per
// window offset.
// CHECK: func.func @maxpool1d
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x3x16xf32>)
// CHECK-DAG: %[[sel1:.*]] = arith.constant dense<{{\[\[\[}}0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00]{{.*}}> : tensor<3x3x2xf32>
// CHECK-DAG: %[[sel0:.*]] = arith.constant dense<{{\[\[\[}}1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]{{.*}}> : tensor<3x3x2xf32>
// CHECK-DAG: %[[zero:.*]] = arith.constant dense<0.000000e+00> : tensor<1x3x8xf32>
// CHECK-DAG: %[[input:.*]] = tensor.extract_slice {{.*}} to tensor<1x3x16xf32>
// CHECK: %[[conv0:.*]] = linalg.conv_1d_ncw_fcw
// CHECK-SAME: strides = dense<2>
// CHECK-SAME: ins(%[[input]], %[[sel0]] : tensor<1x3x16xf32>, tensor<3x3x2xf32>) outs(%[[zero]] : tensor<1x3x8xf32>)
// CHECK: %[[conv1:.*]] = linalg.conv_1d_ncw_fcw
// CHECK-SAME: strides = dense<2>
// CHECK-SAME: ins(%[[input]], %[[sel1]] : tensor<1x3x16xf32>, tensor<3x3x2xf32>) outs(%[[zero]] : tensor<1x3x8xf32>)
// CHECK: arith.maximumf %[[conv0]], %[[conv1]] : tensor<1x3x8xf32>
// CHECK-NOT: linalg.pooling
// CHECK: return
func.func @maxpool1d(%arg0: tensor<1x3x16xf32>) -> tensor<1x3x8xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [1, 3, 1, 16] : tensor<1x3x16xf32> into tensor<1x3x1x16xf32>
  %0 = tensor.empty() : tensor<1x3x1x8xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x3x1x8xf32>) -> tensor<1x3x1x8xf32>
  %2 = tensor.empty() : tensor<1x2xf32>
  %3 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<[1, 2]> : vector<2xi64>} ins(%expanded, %2 : tensor<1x3x1x16xf32>, tensor<1x2xf32>) outs(%1 : tensor<1x3x1x8xf32>) -> tensor<1x3x1x8xf32>
  %collapsed = tensor.collapse_shape %3 [[0], [1, 2], [3]] : tensor<1x3x1x8xf32> into tensor<1x3x8xf32>
  return %collapsed : tensor<1x3x8xf32>
}

// -----

// A native 1D max pool with a window of 3 needs 3 selection convolutions, whose
// maxima are combined in a balanced tree.
// CHECK: func.func @window3
// CHECK: %[[c0:.*]] = linalg.conv_1d_ncw_fcw
// CHECK: %[[c1:.*]] = linalg.conv_1d_ncw_fcw
// CHECK: %[[c2:.*]] = linalg.conv_1d_ncw_fcw
// CHECK: %[[m01:.*]] = arith.maximumf %[[c0]], %[[c1]]
// CHECK: %[[m:.*]] = arith.maximumf %[[m01]], %[[c2]]
// CHECK: return %[[m]]
func.func @window3(%arg0: tensor<1x2x9xf32>) -> tensor<1x2x3xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1x2x3xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  %2 = tensor.empty() : tensor<3xf32>
  %3 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, strides = dense<3> : vector<1xi64>} ins(%arg0, %2 : tensor<1x2x9xf32>, tensor<3xf32>) outs(%1 : tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  return %3 : tensor<1x2x3xf32>
}

// -----

// A max pool whose init is not -inf is left alone.
// CHECK: func.func @nonneg_inf_init
// CHECK: linalg.pooling_ncw_max
// CHECK-NOT: linalg.conv_1d_ncw_fcw
func.func @nonneg_inf_init(%arg0: tensor<1x2x8xf32>) -> tensor<1x2x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x2x4xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  %2 = tensor.empty() : tensor<2xf32>
  %3 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%arg0, %2 : tensor<1x2x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  return %3 : tensor<1x2x4xf32>
}

// -----

// A genuinely 2D max pool is left alone.
// CHECK: func.func @maxpool2d
// CHECK: linalg.pooling_nchw_max
// CHECK-NOT: linalg.conv
func.func @maxpool2d(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x3x8x8xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1x3x8x8xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  %2 = tensor.empty() : tensor<2x2xf32>
  %3 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %2 : tensor<1x3x16x16xf32>, tensor<2x2xf32>) outs(%1 : tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  return %3 : tensor<1x3x8x8xf32>
}
