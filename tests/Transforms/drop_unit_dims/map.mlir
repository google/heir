// RUN: heir-opt %s --drop-unit-dims --canonicalize --split-input-file | FileCheck %s

module{
  // CHECK: func @mulf
  // CHECK-SAME: %[[arg0:.*]]: tensor<1x512xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<1x512xf32>)
  func.func @mulf(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
    %0 = tensor.empty() : tensor<1x512xf32>
    %mapped = linalg.map { arith.mulf } ins(%arg0, %arg1 : tensor<1x512xf32>, tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>)
    // CHECK-DAG: %[[v0:.*]] = tensor.empty() : tensor<1x512xf32>
    // CHECK-DAG: %[[collapsed_alloc:.*]] = tensor.collapse_shape %[[v0]] {{\[\[}}0, 1]] : tensor<1x512xf32> into tensor<512xf32>
    // CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[arg0]] {{\[\[}}0, 1]] : tensor<1x512xf32> into tensor<512xf32>
    // CHECK-DAG: %[[collapsed0:.*]] = tensor.collapse_shape %[[arg1]] {{\[\[}}0, 1]] : tensor<1x512xf32> into tensor<512xf32>
    // CHECK:     %[[mapped:.*]] = linalg.map { arith.mulf } ins(%[[collapsed]], %[[collapsed0]] : tensor<512xf32>, tensor<512xf32>)
    // CHECK-SAME:   outs(%[[collapsed_alloc]] : tensor<512xf32>)
    // CHECK:     %[[expanded:.*]] = tensor.expand_shape %[[mapped]] {{\[\[}}0, 1]] output_shape [1, 512]
    // CHECK:     return %[[expanded]] : tensor<1x512xf32>
    return %mapped : tensor<1x512xf32>
  }
}

// -----

module{
  // CHECK: func @unary
  // CHECK-SAME: %[[arg0:.*]]: tensor<1x512xf32>)
  func.func @unary(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
    %0 = tensor.empty() : tensor<1x512xf32>
    %mapped = linalg.map { math.absf } ins(%arg0 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>)
    // CHECK-DAG: %[[v0:.*]] = tensor.empty() : tensor<1x512xf32>
    // CHECK-DAG: %[[collapsed_alloc:.*]] = tensor.collapse_shape %[[v0]] {{\[\[}}0, 1]] : tensor<1x512xf32> into tensor<512xf32>
    // CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[arg0]] {{\[\[}}0, 1]] : tensor<1x512xf32> into tensor<512xf32>
    // CHECK:     %[[mapped:.*]] = linalg.map { math.absf } ins(%[[collapsed]] : tensor<512xf32>)
    // CHECK-SAME:   outs(%[[collapsed_alloc]] : tensor<512xf32>)
    // CHECK:     %[[expanded:.*]] = tensor.expand_shape %[[mapped]] {{\[\[}}0, 1]] output_shape [1, 512]
    // CHECK:     return %[[expanded]] : tensor<1x512xf32>
    return %mapped : tensor<1x512xf32>
  }
}

// -----

module{
  // CHECK: func @multiple_dims
  // CHECK-SAME: %[[arg0:.*]]: tensor<1x1x512xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<1x1x512xf32>)
  func.func @multiple_dims(%arg0: tensor<1x1x512xf32>, %arg1: tensor<1x1x512xf32>) -> tensor<1x1x512xf32> {
    %0 = tensor.empty() : tensor<1x1x512xf32>
    %mapped = linalg.map { arith.mulf } ins(%arg0, %arg1 : tensor<1x1x512xf32>, tensor<1x1x512xf32>) outs(%0 : tensor<1x1x512xf32>)
    // CHECK-DAG: %[[v0:.*]] = tensor.empty() : tensor<1x1x512xf32>
    // CHECK-DAG: %[[collapsed_alloc:.*]] = tensor.collapse_shape %[[v0]] {{\[\[}}0, 1, 2]] : tensor<1x1x512xf32> into tensor<512xf32>
    // CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[arg0]] {{\[\[}}0, 1, 2]] : tensor<1x1x512xf32> into tensor<512xf32>
    // CHECK-DAG: %[[collapsed0:.*]] = tensor.collapse_shape %[[arg1]] {{\[\[}}0, 1, 2]] : tensor<1x1x512xf32> into tensor<512xf32>
    // CHECK:     %[[mapped:.*]] = linalg.map { arith.mulf } ins(%[[collapsed]], %[[collapsed0]] : tensor<512xf32>, tensor<512xf32>)
    // CHECK-SAME:   outs(%[[collapsed_alloc]] : tensor<512xf32>)
    // CHECK:     %[[expanded:.*]] = tensor.expand_shape %[[mapped]] {{\[\[}}0, 1, 2]] output_shape [1, 1, 512]
    // CHECK:     return %[[expanded]] : tensor<1x1x512xf32>
    return %mapped : tensor<1x1x512xf32>
  }
}

// -----

module{
  // CHECK: func @middle_dims
  // CHECK-SAME: %[[arg0:.*]]: tensor<1x2x1x512xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<1x2x1x512xf32>)
  func.func @middle_dims(%arg0: tensor<1x2x1x512xf32>, %arg1: tensor<1x2x1x512xf32>) -> tensor<1x2x1x512xf32> {
    %0 = tensor.empty() : tensor<1x2x1x512xf32>
    %mapped = linalg.map { arith.mulf } ins(%arg0, %arg1 : tensor<1x2x1x512xf32>, tensor<1x2x1x512xf32>) outs(%0 : tensor<1x2x1x512xf32>)
    // CHECK-DAG: %[[v0:.*]] = tensor.empty() : tensor<1x2x1x512xf32>
    // CHECK-DAG: %[[collapsed_alloc:.*]] = tensor.collapse_shape %[[v0]] {{\[\[}}0, 1, 2], [3]] : tensor<1x2x1x512xf32> into tensor<2x512xf32>
    // CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[arg0]] {{\[\[}}0, 1, 2], [3]] : tensor<1x2x1x512xf32> into tensor<2x512xf32>
    // CHECK-DAG: %[[collapsed0:.*]] = tensor.collapse_shape %[[arg1]] {{\[\[}}0, 1, 2], [3]] : tensor<1x2x1x512xf32> into tensor<2x512xf32>
    // CHECK:     %[[mapped:.*]] = linalg.map { arith.mulf } ins(%[[collapsed]], %[[collapsed0]] : tensor<2x512xf32>, tensor<2x512xf32>)
    // CHECK-SAME:   outs(%[[collapsed_alloc]] : tensor<2x512xf32>)
    // CHECK:     %[[expanded:.*]] = tensor.expand_shape %[[mapped]] {{\[\[}}0, 1, 2], [3]] output_shape [1, 2, 1, 512]
    // CHECK:     return %[[expanded]] : tensor<1x2x1x512xf32>
    return %mapped : tensor<1x2x1x512xf32>
  }
}
