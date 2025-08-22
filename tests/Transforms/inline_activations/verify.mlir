// RUN: heir-opt --inline-activations --split-input-file %s | FileCheck %s

// Doesn't inline because the function is a declaration.
module {
  // CHECK: func.func @main
  // CHECK-NEXT: call @relu(
  // CHECK-NEXT: return
  func.func @main(%arg0: tensor<1x512xf32>) -> (tensor<1x512xf32>) {
    %0 = call @relu(%arg0) : (tensor<1x512xf32>) -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>
  }
  func.func private @relu(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32>
}

// -----

module {
  // CHECK: func.func @main
  // CHECK-NEXT: call @relu_not
  // CHECK-NEXT: return
  func.func @main(%arg0: tensor<1x512xf32>) -> (tensor<1x512xf32>) {
    %0 = call @relu_not(%arg0) : (tensor<1x512xf32>) -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>
  }
  func.func @relu_not(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<1x512xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<f32>) outs(%0 : tensor<1x512xf32>) dimensions = [0, 1]
    %1 = tensor.empty() : tensor<1x512xf32>
    %mapped = linalg.map { arith.maximumf } ins(%arg0, %broadcasted : tensor<1x512xf32>, tensor<1x512xf32>) outs(%1 : tensor<1x512xf32>)
    return %mapped : tensor<1x512xf32>
  }
}
