func.func @dot_product(%arg0: tensor<8xi16> {secret.secret}, %arg1: tensor<8xi16> {secret.secret}) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %arg2 = 0 to 8 iter_args(%iter = %c0_si16) -> (i16) {
    %1 = tensor.extract %arg0[%arg2] : tensor<8xi16>
    %2 = tensor.extract %arg1[%arg2] : tensor<8xi16>
    %3 = arith.muli %1, %2 : i16
    %4 = arith.addi %iter, %3 : i16
    affine.yield %4 : i16
  }
  return %0 : i16
}
