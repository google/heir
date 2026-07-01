// RUN: heir-opt --linalg-canonicalizations --canonicalize --split-input-file %s | FileCheck %s

// Fuses conv_2d_nchw_fchw + pooling_nchw_sum with constant conv outs init.
module {
  // CHECK: func.func @conv_sum_pool_const_outs
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x1x6x6xf32>)
  // CHECK-DAG: %[[FUSED_FILTER:.*]] = arith.constant dense<{{.*}}> : tensor<1x1x4x4xf32>
  // CHECK-DAG: %[[CONV_INIT:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1x3x3xf32>
  // CHECK-DAG: %[[POOL_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK-DAG: %[[POOL_KERNEL:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK: %[[POOLED_INIT:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: ins(%[[CONV_INIT]], %[[POOL_KERNEL]] : tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>)
  // CHECK-SAME: outs(%[[POOL_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: %[[RES:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME: strides = dense<2> : tensor<2xi64>
  // CHECK-SAME: ins(%[[INPUT]], %[[FUSED_FILTER]] : tensor<1x1x6x6xf32>, tensor<1x1x4x4xf32>)
  // CHECK-SAME: outs(%[[POOLED_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: return %[[RES]]
  func.func @conv_sum_pool_const_outs(%input: tensor<1x1x6x6xf32>) -> tensor<1x1x2x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %conv_init = arith.constant dense<1.000000e+00> : tensor<1x1x3x3xf32>

    // Conv filter: 2x2
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>

    // Conv: stride 2
    %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x1x6x6xf32>, tensor<1x1x2x2xf32>)
      outs(%conv_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

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

// Fuses conv_2d_nchw_fchw + pooling_nchw_sum with argument conv outs init.
module {
  // CHECK: func.func @conv_sum_pool_arg_outs
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x1x6x6xf32>, %[[CONV_INIT:.*]]: tensor<1x1x3x3xf32>)
  // CHECK-DAG: %[[FUSED_FILTER:.*]] = arith.constant dense<{{.*}}> : tensor<1x1x4x4xf32>
  // CHECK-DAG: %[[POOL_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK-DAG: %[[POOL_KERNEL:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK: %[[POOLED_INIT:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: ins(%[[CONV_INIT]], %[[POOL_KERNEL]] : tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>)
  // CHECK-SAME: outs(%[[POOL_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: %[[RES:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME: strides = dense<2> : tensor<2xi64>
  // CHECK-SAME: ins(%[[INPUT]], %[[FUSED_FILTER]] : tensor<1x1x6x6xf32>, tensor<1x1x4x4xf32>)
  // CHECK-SAME: outs(%[[POOLED_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: return %[[RES]]
  func.func @conv_sum_pool_arg_outs(%input: tensor<1x1x6x6xf32>, %conv_init: tensor<1x1x3x3xf32>) -> tensor<1x1x2x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Conv filter: 2x2
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>

    // Conv: stride 2
    %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x1x6x6xf32>, tensor<1x1x2x2xf32>)
      outs(%conv_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

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

// Fuses conv_2d_nchw_fchw + pooling_nchw_sum + div (average pooling) with constant conv outs init.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @conv_avg_pool_const_outs
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x1x6x6xf32>)
  // CHECK-DAG: %[[FUSED_FILTER:.*]] = arith.constant dense<{{.*}}> : tensor<1x1x4x4xf32>
  // CHECK-DAG: %[[CONV_INIT:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1x3x3xf32>
  // CHECK-DAG: %[[POOL_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1x2x2xf32>
  // CHECK-DAG: %[[POOL_KERNEL:.*]] = arith.constant dense<2.500000e-01> : tensor<1x1x2x2xf32>
  // CHECK: %[[POOLED_INIT:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: ins(%[[CONV_INIT]], %[[POOL_KERNEL]] : tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>)
  // CHECK-SAME: outs(%[[POOL_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: %[[RES:.*]] = linalg.conv_2d_nchw_fchw
  // CHECK-SAME: dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME: strides = dense<2> : tensor<2xi64>
  // CHECK-SAME: ins(%[[INPUT]], %[[FUSED_FILTER]] : tensor<1x1x6x6xf32>, tensor<1x1x4x4xf32>)
  // CHECK-SAME: outs(%[[POOLED_INIT]] : tensor<1x1x2x2xf32>)
  // CHECK: return %[[RES]]
  func.func @conv_avg_pool_const_outs(%input: tensor<1x1x6x6xf32>) -> tensor<1x1x2x2xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %div_cst = arith.constant 4.000000e+00 : f32
    %conv_init = arith.constant dense<1.000000e+00> : tensor<1x1x3x3xf32>

    // Conv filter: 2x2
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>

    // Conv: stride 2
    %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x1x6x6xf32>, tensor<1x1x2x2xf32>)
      outs(%conv_init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

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
