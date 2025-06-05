// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

module {
  // CHECK: func @main
  // CHECK: %[[cst:.*]] = arith.constant dense<1.{{0*}}e+00> : tensor<512xf32>
  // CHECK: return %[[cst]] : tensor<512xf32>
  func.func @main() -> (tensor<512xf32>) {
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<f32>
    %1 = tensor.empty() : tensor<512xf32>
    %broadcasted = linalg.broadcast ins(%cst_0 : tensor<f32>) outs(%1 : tensor<512xf32>) dimensions = [0]
    func.return %broadcasted : tensor<512xf32>
  }
}

// -----

module {
  // CHECK: func @multidim
  // CHECK: %[[cst:.*]] = arith.constant dense<1.{{0*}}e+00> : tensor<1x512xf32>
  // CHECK: return %[[cst]] : tensor<1x512xf32>
  func.func @multidim() -> (tensor<1x512xf32>) {
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<f32>
    %1 = tensor.empty() : tensor<1x512xf32>
    %broadcasted = linalg.broadcast ins(%cst_0 : tensor<f32>) outs(%1 : tensor<1x512xf32>) dimensions = [0, 1]
    func.return %broadcasted : tensor<1x512xf32>
  }
}
