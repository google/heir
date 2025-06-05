// RUN: heir-opt --linalg-canonicalizations %s | FileCheck %s

module {
  // CHECK: func @main
  // CHECK: %[[cst:.*]] = arith.constant dense<1.{{0*}}e+00> : tensor<1x512xf32>
  // CHECK: return %[[cst]] : tensor<1x512xf32>
  func.func @main() -> (tensor<1x512xf32>) {
    %cst = arith.constant 1.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x512xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x512xf32>) -> tensor<1x512xf32>
    func.return %4 : tensor<1x512xf32>
  }
}
