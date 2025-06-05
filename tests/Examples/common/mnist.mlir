// This file contains a minimal torch-exported (and canonicalized) mnist model
// composed of feedforward layers and ReLU activation. The export path is torch
// -> stablehlo -> mlir.

module {
  func.func @main(
      %layer1_weights: tensor<512x784xf32> {debug.name = "layer1_weights"},
      %layer1_bias: tensor<512xf32> {debug.name = "layer1_bias"},
      %layer2_weights: tensor<10x512xf32> {debug.name = "layer2_weights"},
      %layer2_bias: tensor<10xf32> {debug.name="layer2_bias"},
      %input: tensor<1x784xf32> {secret.secret, debug.name = "input"}
  ) -> (tensor<1x10xf32> {debug.name = "result"}) {
    %c0_10 = arith.constant dense<0.000000e+00> : tensor<10xf32>
    %c0_512 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = tensor.empty() : tensor<784x512xf32>
    %transposed = linalg.transpose ins(%layer1_weights : tensor<512x784xf32>) outs(%0 : tensor<784x512xf32>) permutation = [1, 0]
    %collapsed = tensor.collapse_shape %input [[0, 1]] : tensor<1x784xf32> into tensor<784xf32>
    %1 = linalg.vecmat ins(%collapsed, %transposed : tensor<784xf32>, tensor<784x512xf32>) outs(%c0_512 : tensor<512xf32>) -> tensor<512xf32>
    %2 = arith.addf %layer1_bias, %1 : tensor<512xf32>
    %3 = arith.maximumf %2, %c0_512 : tensor<512xf32>
    %4 = tensor.empty() : tensor<512x10xf32>
    %transposed_1 = linalg.transpose ins(%layer2_weights : tensor<10x512xf32>) outs(%4 : tensor<512x10xf32>) permutation = [1, 0]
    %5 = linalg.vecmat ins(%3, %transposed_1 : tensor<512xf32>, tensor<512x10xf32>) outs(%c0_10 : tensor<10xf32>) -> tensor<10xf32>
    %6 = arith.addf %layer2_bias, %5 : tensor<10xf32>
    %expanded = tensor.expand_shape %6 [[0, 1]] output_shape [1, 10] : tensor<10xf32> into tensor<1x10xf32>
    return %expanded : tensor<1x10xf32>
  }
}
