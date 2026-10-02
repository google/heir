// RUN: heir-opt --broadcast-swap --split-input-file %s | FileCheck %s
// This tests part of a layernorm like operation.

module {
  // CHECK: func @main
  func.func @main(%arg0: tensor<2xf32> {secret.secret}) -> tensor<2xf32> {
    %cst = arith.constant dense<2.00> : tensor<f32>
    // CHECK: linalg.reduce ins(%arg0 : tensor<2xf32>) outs(%cst : tensor<f32>) dimensions = [0]
    %reduced = linalg.reduce ins(%arg0 : tensor<2xf32>) outs(%cst : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %11 = arith.addf %in, %init : f32
        linalg.yield %11 : f32
      }
    // CHECK: linalg.broadcast ins(%reduced : tensor<f32>) outs(%0 : tensor<2xf32>) dimensions = [0]
    %0 = arith.mulf %reduced, %cst : tensor<f32>
    // CHECK: %1 = arith.mulf %broadcasted, %cst_0 : tensor<2xf32>
    // CHECK: arith.subf %arg0, %1 : tensor<2xf32>
    %1 = tensor.empty() : tensor<2xf32>
    %broadcasted = linalg.broadcast ins(%0 : tensor<f32>) outs(%1 : tensor<2xf32>) dimensions = [0]
    %2 = arith.subf %arg0, %broadcasted : tensor<2xf32>
    %3 = arith.mulf %2, %2 : tensor<2xf32>
    return %3 : tensor<2xf32>
  }
}
