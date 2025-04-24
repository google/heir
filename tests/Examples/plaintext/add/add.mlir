func.func @add(%arg0: tensor<8xi16> {secret.secret}) -> i16 {
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %arg0[%c0] : tensor<8xi16>
  return %1 : i16
}
