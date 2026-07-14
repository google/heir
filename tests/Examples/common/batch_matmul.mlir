func.func @batch_matmul(%arg0: tensor<2x17x19xf32> {secret.secret}, %arg1: tensor<2x19x21xf32> {secret.secret}) -> tensor<2x17x21xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x17x21xf32>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<2x17x19xf32>, tensor<2x19x21xf32>) outs(%cst : tensor<2x17x21xf32>) -> tensor<2x17x21xf32>
  return %2 : tensor<2x17x21xf32>
}
