func.func @pad_matmul(%arg0: tensor<5x5xf32> {secret.secret}, %arg1: tensor<5x5xf32> {secret.secret}) -> tensor<5x7xf32> {
  %cst = arith.constant dense<0.0> : tensor<5x7xf32>
  %c0 = arith.constant 0.0 : f32

  %padded_a = tensor.pad %arg0 low[0, 0] high[0, 1] {
  ^body(%arg2: index, %arg3: index):
    tensor.yield %c0 : f32
  } : tensor<5x5xf32> to tensor<5x6xf32>

  %padded_b = tensor.pad %arg1 low[1, 2] high[0, 0] {
  ^body(%arg2: index, %arg3: index):
    tensor.yield %c0 : f32
  } : tensor<5x5xf32> to tensor<6x7xf32>

  %0 = linalg.matmul ins(%padded_a, %padded_b : tensor<5x6xf32>, tensor<6x7xf32>) outs(%cst : tensor<5x7xf32>) -> tensor<5x7xf32>
  return %0 : tensor<5x7xf32>
}
