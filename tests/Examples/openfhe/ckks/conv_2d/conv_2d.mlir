module {
  func.func @conv_2d(%arg0: !secret.secret<tensor<4x4xf32>>, %arg1: tensor<3x3xf32>) -> !secret.secret<tensor<2x2xf32>> {
    %1 = tensor.empty() : tensor<2x2xf32>
    %3 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>>) {
    ^body(%input0: tensor<4x4xf32>):
      %4 = linalg.conv_2d ins(%input0, %arg1 : tensor<4x4xf32>, tensor<3x3xf32>) outs(%1 : tensor<2x2xf32>) -> tensor<2x2xf32>
      secret.yield %4 : tensor<2x2xf32>
    } -> !secret.secret<tensor<2x2xf32>>
    return %3 : !secret.secret<tensor<2x2xf32>>
  }
}
