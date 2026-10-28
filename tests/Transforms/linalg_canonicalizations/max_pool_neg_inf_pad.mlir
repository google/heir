// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// A ReLU output padded with -inf for a max pool (torch's MaxPool2d(3, 2, 1)):
// every window holds an input element, so the pad becomes zero.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @relu_max_pool
// CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: tensor.pad
// CHECK: tensor.yield %[[zero]] : f32
// CHECK: linalg.pooling_nchw_max
func.func @relu_max_pool(%x: tensor<1x2x8x8xf32>) -> tensor<1x2x4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %neg_inf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<1x2x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x2x8x8xf32>) outs(%empty : tensor<1x2x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %zero : f32
    %s = arith.select %c, %in, %zero : f32
    linalg.yield %s : f32
  } -> tensor<1x2x8x8xf32>
  %padded = tensor.pad %relu low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%i: index, %j: index, %k: index, %l: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x2x8x8xf32> to tensor<1x2x10x10xf32>
  %init = tensor.empty() : tensor<1x2x4x4xf32>
  %filled = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %pool = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x2x10x10xf32>, tensor<3x3xf32>) outs(%filled : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %pool : tensor<1x2x4x4xf32>
}

// -----

// An input that may be negative keeps its -inf padding.

// CHECK: func.func @max_pool
// CHECK: %[[neg_inf:.*]] = arith.constant 0xFF800000 : f32
// CHECK: tensor.yield %[[neg_inf]] : f32
func.func @max_pool(%x: tensor<1x2x8x8xf32>) -> tensor<1x2x4x4xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%i: index, %j: index, %k: index, %l: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x2x8x8xf32> to tensor<1x2x10x10xf32>
  %init = tensor.empty() : tensor<1x2x4x4xf32>
  %filled = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %pool = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x2x10x10xf32>, tensor<3x3xf32>) outs(%filled : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %pool : tensor<1x2x4x4xf32>
}

// -----

// With dilation 3 the 2x2 window's taps, at 0 and 3 in the padded input, both
// skip the single input element, so the window holds only padding and the
// pad stays -inf.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @dilated_max_pool
// CHECK: %[[neg_inf:.*]] = arith.constant 0xFF800000 : f32
// CHECK: tensor.pad
// CHECK: tensor.yield %[[neg_inf]] : f32
func.func @dilated_max_pool(%x: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
  %zero = arith.constant 0.0 : f32
  %neg_inf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<1x1x1x1xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x1x1x1xf32>) outs(%empty : tensor<1x1x1x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %zero : f32
    %s = arith.select %c, %in, %zero : f32
    linalg.yield %s : f32
  } -> tensor<1x1x1x1xf32>
  %padded = tensor.pad %relu low[0, 0, 1, 1] high[0, 0, 2, 2] {
  ^bb0(%i: index, %j: index, %k: index, %l: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x1x1x1xf32> to tensor<1x1x4x4xf32>
  %window = tensor.empty() : tensor<2x2xf32>
  %init = tensor.empty() : tensor<1x1x1x1xf32>
  %filled = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %pool = linalg.pooling_nchw_max {dilations = dense<3> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %window : tensor<1x1x4x4xf32>, tensor<2x2xf32>) outs(%filled : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  return %pool : tensor<1x1x1x1xf32>
}
