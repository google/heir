func.func @pad_dot(%arg0: tensor<5xf32> {secret.secret}, %arg1: tensor<8xf32> {secret.secret}) -> f32 {
  %cst = arith.constant dense<0.0> : tensor<f32>
  %c0 = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[2] high[1] {
  ^body(%arg2: index):
    tensor.yield %c0 : f32
  } : tensor<5xf32> to tensor<8xf32>
  %0 = linalg.dot ins(%padded, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%cst : tensor<f32>) -> tensor<f32>
  %1 = tensor.extract %0[] : tensor<f32>
  return %1 : f32
}
