func.func @matmul_secret_secret(%arg0: tensor<3x5xf32> {secret.secret}, %arg1: tensor<5x2xf32> {secret.secret}) -> tensor<3x2xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x2xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<3x5xf32>, tensor<5x2xf32>) outs(%cst : tensor<3x2xf32>) -> tensor<3x2xf32>
  return %2 : tensor<3x2xf32>
}
