func.func @bicyclic_matmul_pt(%arg0: tensor<13x14xf32> {secret.secret}, %arg1: tensor<14x16xf32>) -> tensor<13x16xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<13x16xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<13x14xf32>, tensor<14x16xf32>) outs(%cst : tensor<13x16xf32>) -> tensor<13x16xf32>
  return %2 : tensor<13x16xf32>
}
