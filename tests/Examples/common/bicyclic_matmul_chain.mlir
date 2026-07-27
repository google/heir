func.func @bicyclic_matmul_chain(%arg0: tensor<13x18xf32> {secret.secret}, %arg1: tensor<18x16xf32>, %arg2: tensor<16x9xf32> {secret.secret}) -> tensor<13x9xf32> {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<13x16xf32>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<13x9xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<13x18xf32>, tensor<18x16xf32>) outs(%cst0 : tensor<13x16xf32>) -> tensor<13x16xf32>
  %1 = linalg.matmul ins(%0, %arg2 : tensor<13x16xf32>, tensor<16x9xf32>) outs(%cst1 : tensor<13x9xf32>) -> tensor<13x9xf32>
  return %1 : tensor<13x9xf32>
}
