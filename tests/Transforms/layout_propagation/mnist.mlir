// RUN: heir-opt --mlir-print-local-scope --layout-propagation %s | FileCheck %s

module attributes {backend.openfhe, scheme.bgv} {
  // CHECK: @main
  func.func @main(%arg0: tensor<512x784xf32> {debug.name = "layer1_weights"}, %arg1: tensor<512xf32> {debug.name = "layer1_bias"}, %arg2: tensor<10x512xf32> {debug.name = "layer2_weights"}, %arg3: tensor<10xf32> {debug.name = "layer2_bias"}, %arg4: !secret.secret<tensor<1x784xf32>> {debug.name = "input"}) -> (!secret.secret<tensor<1x10xf32>> {debug.name = "result"}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<10xf32>
    %0 = tensor.empty() : tensor<784x512xf32>
    %1 = tensor.empty() : tensor<512x10xf32>
    %2 = secret.generic(%arg4: !secret.secret<tensor<1x784xf32>>) {
    ^body(%input0: tensor<1x784xf32>):
      %transposed = linalg.transpose ins(%arg0 : tensor<512x784xf32>) outs(%0 : tensor<784x512xf32>) permutation = [1, 0]
      // CHECK: tensor.collapse_shape
      // CHECK-SAME: {tensor_ext.layout = #tensor_ext.layout<"{ [i0] ->
      %collapsed = tensor.collapse_shape %input0 [[0, 1]] : tensor<1x784xf32> into tensor<784xf32>
      %3 = linalg.vecmat ins(%collapsed, %transposed : tensor<784xf32>, tensor<784x512xf32>) outs(%cst : tensor<512xf32>) -> tensor<512xf32>
      %4 = arith.addf %arg1, %3 : tensor<512xf32>
      %5 = arith.maximumf %4, %cst : tensor<512xf32>
      %transposed_1 = linalg.transpose ins(%arg2 : tensor<10x512xf32>) outs(%1 : tensor<512x10xf32>) permutation = [1, 0]
      %6 = linalg.vecmat ins(%5, %transposed_1 : tensor<512xf32>, tensor<512x10xf32>) outs(%cst_0 : tensor<10xf32>) -> tensor<10xf32>
      %7 = arith.addf %arg3, %6 : tensor<10xf32>
      // CHECK: tensor.expand_shape
      // CHECK-SAME: {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] ->
      %expanded = tensor.expand_shape %7 [[0, 1]] output_shape [1, 10] : tensor<10xf32> into tensor<1x10xf32>
      secret.yield %expanded : tensor<1x10xf32>
    } -> !secret.secret<tensor<1x10xf32>>
    return %2 : !secret.secret<tensor<1x10xf32>>
  }
}
