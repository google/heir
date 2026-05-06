func.func @dot_product(%arg0: tensor<1024xf32> {secret.secret}, %arg1: tensor<1024xf32> {secret.secret}) -> (tensor<f32> {secret.secret}) {
  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = linalg.dot {secret.secret} ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%cst : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
