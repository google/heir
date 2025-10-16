// RUN: heir-opt --layout-propagation %s | FileCheck %s

// An MNIST-like two-layer neural network with cleartext weights and biases
// and a secret input.

// CHECK: @main
func.func @main(
    %arg0: tensor<512x784xf32>,
    %arg1: tensor<512xf32>,
    %arg2: tensor<10x512xf32>,
    %arg3: tensor<10xf32>,
    %arg4: !secret.secret<tensor<784xf32>>) -> !secret.secret<tensor<10xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<10xf32>
  %0 = tensor.empty() : tensor<784x512xf32>
  %1 = tensor.empty() : tensor<512x10xf32>
  %2 = secret.generic(%arg4: !secret.secret<tensor<784xf32>>) {
  ^body(%input0: tensor<784xf32>):
    %transposed = linalg.transpose ins(%arg0 : tensor<512x784xf32>) outs(%0 : tensor<784x512xf32>) permutation = [1, 0]
    %3 = linalg.vecmat ins(%input0, %transposed : tensor<784xf32>, tensor<784x512xf32>) outs(%cst : tensor<512xf32>) -> tensor<512xf32>
    %4 = arith.addf %arg1, %3 : tensor<512xf32>
    %5 = arith.maximumf %4, %cst : tensor<512xf32>
    %transposed_1 = linalg.transpose ins(%arg2 : tensor<10x512xf32>) outs(%1 : tensor<512x10xf32>) permutation = [1, 0]
    %6 = linalg.vecmat ins(%5, %transposed_1 : tensor<512xf32>, tensor<512x10xf32>) outs(%cst_0 : tensor<10xf32>) -> tensor<10xf32>
    %7 = arith.addf %arg3, %6 : tensor<10xf32>
    secret.yield %7 : tensor<10xf32>
  } -> !secret.secret<tensor<10xf32>>
  return %2 : !secret.secret<tensor<10xf32>>
}
