func.func @bicyclic_matmul(%arg0: tensor<16x17xf32> {secret.secret}, %arg1: tensor<17x19xf32> {secret.secret}) -> tensor<16x19xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x19xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<16x17xf32>, tensor<17x19xf32>) outs(%cst : tensor<16x19xf32>) -> tensor<16x19xf32>
  return %2 : tensor<16x19xf32>
}
