func.func @main(%input: tensor<25xi32> {secret.secret}, %mat: tensor<15x25xi32>) -> tensor<15xi32> {
  %cst = arith.constant dense<0> : tensor<15xi32>
  %1 = linalg.matvec ins(%mat, %input : tensor<15x25xi32>, tensor<25xi32>) outs(%cst : tensor<15xi32>) -> tensor<15xi32>
  return %1 : tensor<15xi32>
}
