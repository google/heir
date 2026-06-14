// A comment to force rebuild
func.func @dot_product(%arg0: tensor<8xf32> {secret.secret}, %arg1: tensor<8xf32> {secret.secret}) -> f32 {
  %cst = arith.constant dense<1.000000e-01> : tensor<f32>
  %0 = linalg.dot ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%cst : tensor<f32>) -> tensor<f32>
  %1 = tensor.extract %0[] : tensor<f32>
  return %1 : f32
}
