module {
  func.func @conv_1d(%arg0: !secret.secret<tensor<9xf32>>, %arg1: tensor<4xf32>) -> !secret.secret<tensor<5xf32>> {
    %1 = tensor.empty() : tensor<5xf32>
    %3 = secret.generic(%arg0: !secret.secret<tensor<9xf32>>) {
    ^body(%input0: tensor<9xf32>):
      %4 = linalg.conv_1d ins(%input0, %arg1 : tensor<9xf32>, tensor<4xf32>) outs(%1 : tensor<5xf32>) -> tensor<5xf32>
      secret.yield %4 : tensor<5xf32>
    } -> !secret.secret<tensor<5xf32>>
    return %3 : !secret.secret<tensor<5xf32>>
  }
}
