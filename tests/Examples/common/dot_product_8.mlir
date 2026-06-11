func.func @dot_product(%arg0: tensor<8xi16> {secret.secret}, %arg1: tensor<8xi16> {secret.secret}) -> i16 {
  %cst = arith.constant dense<0> : tensor<i16>
  %0 = linalg.dot ins(%arg0, %arg1 : tensor<8xi16>, tensor<8xi16>) outs(%cst : tensor<i16>) -> tensor<i16>
  %1 = tensor.extract %0[] : tensor<i16>
  return %1 : i16
}
