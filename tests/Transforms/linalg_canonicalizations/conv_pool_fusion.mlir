// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// Fuses conv_2d_nchw_fchw + pooling_nchw_sum
// Conv: stride 2, filter 2x2
// Pool: stride 1, window 2x2
// Effective stride: 2 * 1 = 2
// Effective filter: 4x4
module {
  // CHECK: func.func @conv_sum_pool
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x1x6x6xf32>)
  // CHECK-DAG: %[[FUSED_FILTER:.*]] = arith.constant dense<{{.*}}1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00{{.*}}3.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00{{.*}}1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00{{.*}}3.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00{{.*}}> : tensor<1x1x4x4xf32>
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK: %[[POOLED_INIT:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: strides = dense<1> : vector<2xi64>
  // CHECK: %[[RES:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME: strides = dense<2> : tensor<2xi64>
  // CHECK-SAME: ins(%[[INPUT]], %[[FUSED_FILTER]] : tensor<1x1x6x6xf32>, tensor<1x1x4x4xf32>)
  // CHECK-SAME: outs(%[[POOLED_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: return %[[RES]]
  func.func @conv_sum_pool(%input: tensor<1x1x6x6xf32>) -> tensor<1x1x2x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Conv output init
    %conv_init = tensor.empty() : tensor<1x1x3x3xf32>
    %conv_fill = linalg.fill ins(%cst_0 : f32) outs(%conv_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // Conv filter: 2x2
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>

    // Conv: stride 2
    %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x1x6x6xf32>, tensor<1x1x2x2xf32>)
      outs(%conv_fill : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // Pool output init
    %pool_init = tensor.empty() : tensor<1x1x2x2xf32>
    %pool_fill = linalg.fill ins(%cst_0 : f32) outs(%pool_init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

    %pool_shape = arith.constant dense<0.0> : tensor<2x2xf32>

    // Pool: stride 1, window 2x2
    %pool = linalg.pooling_nchw_sum
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%conv, %pool_shape : tensor<1x1x3x3xf32>, tensor<2x2xf32>)
      outs(%pool_fill : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

    return %pool : tensor<1x1x2x2xf32>
  }
}

// -----

// Fuses conv_2d_nchw_fchw + pooling_nchw_sum + div (average pooling)
// Conv: stride 2, filter 2x2
// Pool: stride 1, window 2x2
// Div: 4.0
// Effective stride: 2
// Effective filter: 4x4, scaled by 0.25
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @conv_avg_pool
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x1x6x6xf32>)
  // CHECK-DAG: %[[FUSED_FILTER:.*]] = arith.constant dense<{{.*}}2.500000e-01, 5.000000e-01, 2.500000e-01, 5.000000e-01{{.*}}7.500000e-01, 1.000000e+00, 7.500000e-01, 1.000000e+00{{.*}}2.500000e-01, 5.000000e-01, 2.500000e-01, 5.000000e-01{{.*}}7.500000e-01, 1.000000e+00, 7.500000e-01, 1.000000e+00{{.*}}> : tensor<1x1x4x4xf32>
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK: %[[POOLED_INIT:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: strides = dense<1> : vector<2xi64>
  // CHECK: %[[RES:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME: strides = dense<2> : tensor<2xi64>
  // CHECK-SAME: ins(%[[INPUT]], %[[FUSED_FILTER]] : tensor<1x1x6x6xf32>, tensor<1x1x4x4xf32>)
  // CHECK-SAME: outs(%[[POOLED_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: return %[[RES]]
  func.func @conv_avg_pool(%input: tensor<1x1x6x6xf32>) -> tensor<1x1x2x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %div_cst = arith.constant 4.000000e+00 : f32

    // Conv output init
    %conv_init = tensor.empty() : tensor<1x1x3x3xf32>
    %conv_fill = linalg.fill ins(%cst_0 : f32) outs(%conv_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // Conv filter: 2x2
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>

    // Conv: stride 2
    %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x1x6x6xf32>, tensor<1x1x2x2xf32>)
      outs(%conv_fill : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // Pool output init
    %pool_init = tensor.empty() : tensor<1x1x2x2xf32>
    %pool_fill = linalg.fill ins(%cst_0 : f32) outs(%pool_init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

    %pool_shape = arith.constant dense<0.0> : tensor<2x2xf32>

    // Pool: stride 1, window 2x2
    %pool = linalg.pooling_nchw_sum
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%conv, %pool_shape : tensor<1x1x3x3xf32>, tensor<2x2xf32>)
      outs(%pool_fill : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

    // Div
    %div_out = tensor.empty() : tensor<1x1x2x2xf32>
    %div = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%pool : tensor<1x1x2x2xf32>)
      outs(%div_out : tensor<1x1x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %res = arith.divf %in, %div_cst : f32
      linalg.yield %res : f32
    } -> tensor<1x1x2x2xf32>

    return %div : tensor<1x1x2x2xf32>
  }
}

// -----

// Fuses conv_2d_nchw_fchw + pooling_nchw_sum where conv is preceded by tensor.pad.
// The fused conv should directly use the padded tensor.
module {
  // CHECK: func.func @conv_pad_pool
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x1x4x4xf32>)
  // CHECK-DAG: %[[PAD:.*]] = tensor.pad %[[INPUT]]
  // CHECK-DAG: %[[FUSED_FILTER:.*]] = arith.constant dense<{{.*}}> : tensor<1x1x4x4xf32>
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK: %[[POOLED_INIT:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: strides = dense<1> : vector<2xi64>
  // CHECK: %[[RES:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: ins(%[[PAD]], %[[FUSED_FILTER]] : tensor<1x1x6x6xf32>, tensor<1x1x4x4xf32>)
  // CHECK-SAME: outs(%[[POOLED_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: return %[[RES]]
  func.func @conv_pad_pool(%input: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Pad input from 4x4 to 6x6
    %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x1x4x4xf32> to tensor<1x1x6x6xf32>

    // Conv output init
    %conv_init = tensor.empty() : tensor<1x1x3x3xf32>
    %conv_fill = linalg.fill ins(%cst_0 : f32) outs(%conv_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // Conv filter: 2x2
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>

    // Conv: stride 2
    %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%padded, %filter : tensor<1x1x6x6xf32>, tensor<1x1x2x2xf32>)
      outs(%conv_fill : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // Pool output init
    %pool_init = tensor.empty() : tensor<1x1x2x2xf32>
    %pool_fill = linalg.fill ins(%cst_0 : f32) outs(%pool_init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

    %pool_shape = arith.constant dense<0.0> : tensor<2x2xf32>

    // Pool: stride 1, window 2x2
    %pool = linalg.pooling_nchw_sum
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%conv, %pool_shape : tensor<1x1x3x3xf32>, tensor<2x2xf32>)
      outs(%pool_fill : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

    return %pool : tensor<1x1x2x2xf32>
  }
}

// -----

// Negative Test: Intermediate activation layer (conv -> generic -> pool)
// Should NOT fuse.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @no_fuse_activation
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK: math.tanh
  // CHECK: linalg.conv_2d_nchw_fchw
  func.func @no_fuse_activation(%input: tensor<1x1x6x6xf32>) -> tensor<1x1x2x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Conv
    %conv_init = tensor.empty() : tensor<1x1x3x3xf32>
    %conv_fill = linalg.fill ins(%cst_0 : f32) outs(%conv_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>
    %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x1x6x6xf32>, tensor<1x1x2x2xf32>)
      outs(%conv_fill : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // Activation (mocked as a generic copy/scale)
    %act_out = tensor.empty() : tensor<1x1x3x3xf32>
    %act = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv : tensor<1x1x3x3xf32>)
      outs(%act_out : tensor<1x1x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      // just a dummy activation
      %t = math.tanh %in : f32
      linalg.yield %t : f32
    } -> tensor<1x1x3x3xf32>

    // Pool
    %pool_init = tensor.empty() : tensor<1x1x2x2xf32>
    %pool_fill = linalg.fill ins(%cst_0 : f32) outs(%pool_init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %pool_shape = arith.constant dense<0.0> : tensor<2x2xf32>
    %pool = linalg.pooling_nchw_sum
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%act, %pool_shape : tensor<1x1x3x3xf32>, tensor<2x2xf32>)
      outs(%pool_fill : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

    return %pool : tensor<1x1x2x2xf32>
  }
}

// -----

// Negative Test: Pooling has padding (conv -> pad -> pool)
// Should NOT fuse.
module {
  // CHECK: func.func @no_fuse_padded_pool
  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK: tensor.pad
  // CHECK: linalg.conv_2d_nchw_fchw
  func.func @no_fuse_padded_pool(%input: tensor<1x1x6x6xf32>) -> tensor<1x1x3x3xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Conv
    %conv_init = tensor.empty() : tensor<1x1x3x3xf32>
    %conv_fill = linalg.fill ins(%cst_0 : f32) outs(%conv_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>
    %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x1x6x6xf32>, tensor<1x1x2x2xf32>)
      outs(%conv_fill : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    // Pad the conv output before pooling
    // Conv output is 3x3. Pad to 5x5 to allow 3x3 pool with stride 1?
    // Let's pad by 1 on each side: 3x3 -> 5x5
    %padded_conv = tensor.pad %conv low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x1x3x3xf32> to tensor<1x1x5x5xf32>

    // Pool: stride 1, window 3x3
    // Input: 5x5. Window: 3x3. Stride: 1.
    // Output: (5-3)/1 + 1 = 3. Output shape: 3x3.
    %pool_init = tensor.empty() : tensor<1x1x3x3xf32>
    %pool_fill = linalg.fill ins(%cst_0 : f32) outs(%pool_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %pool_shape = arith.constant dense<0.0> : tensor<3x3xf32>
    %pool = linalg.pooling_nchw_sum
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded_conv, %pool_shape : tensor<1x1x5x5xf32>, tensor<3x3xf32>)
      outs(%pool_fill : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

    return %pool : tensor<1x1x3x3xf32>
  }
}
