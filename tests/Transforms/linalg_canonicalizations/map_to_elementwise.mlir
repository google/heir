// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

module {
  // CHECK: func @mulf
  // CHECK-SAME: %[[arg0:.*]]: tensor<1x512xf32>
  // CHECK: %[[cst:.*]] = arith.constant dense<2.{{0*}}e+00> : tensor<1x512xf32>
  // CHECK: %[[v0:.*]] = arith.mulf %[[arg0]], %[[cst]] : tensor<1x512xf32>
  // CHECK: return %[[v0]] : tensor<1x512xf32>
  func.func @mulf(%arg0: tensor<1x512xf32>) -> (tensor<1x512xf32>) {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x512xf32>
    %0 = tensor.empty() : tensor<1x512xf32>
    %mapped = linalg.map { arith.mulf } ins(%cst, %arg0 : tensor<1x512xf32>, tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>)
    func.return %mapped : tensor<1x512xf32>
  }
}

// -----

module {
  // CHECK: func @unary
  // CHECK-SAME: %[[arg0:.*]]: tensor<1xf32>
  // CHECK: %[[v0:.*]] = math.absf %[[arg0]] : tensor<1xf32>
  // CHECK: return %[[v0]] : tensor<1xf32>
  func.func @unary(%arg0: tensor<1xf32>) -> (tensor<1xf32>) {
    %0 = tensor.empty() : tensor<1xf32>
    %mapped = linalg.map { math.absf } ins(%arg0 : tensor<1xf32>) outs(%0 : tensor<1xf32>)
    func.return %mapped : tensor<1xf32>
  }
}

// -----

module {
  // CHECK: func @out_param
  // CHECK: linalg.map
  // CHECK: return
  func.func @out_param(%arg : tensor<?xf32>, %init : tensor<?xf32>) -> tensor<?xf32> attributes {fusion} {
    %0 = linalg.map { math.absf } ins(%arg : tensor<?xf32>) outs(%init : tensor<?xf32>)
    func.return %0 : tensor<?xf32>
  }
}

// -----

// Tests multiple uses of dest tensor.
module{
  // CHECK: func @multiple_dest
  // CHECK-SAME: %[[arg0:.*]]: tensor<512xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<512xf32>)
  // CHECK: %[[v0:.*]] = arith.mulf %[[arg0]]
  // CHECK: %[[v1:.*]] = arith.addf %[[arg1]], %[[v0]]
  // CHECK: return %[[v1]] : tensor<512xf32>
  func.func @multiple_dest(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> (tensor<512xf32>) {
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<512xf32>
    %1 = tensor.empty() : tensor<512xf32>
    %mapped = linalg.map { arith.mulf } ins(%cst_0, %arg0 : tensor<512xf32>, tensor<512xf32>) outs(%1 : tensor<512xf32>)
    %mapped_3 = linalg.map { arith.addf } ins(%arg1, %mapped : tensor<512xf32>, tensor<512xf32>) outs(%1 : tensor<512xf32>)
    return %mapped_3 : tensor<512xf32>
  }
}
