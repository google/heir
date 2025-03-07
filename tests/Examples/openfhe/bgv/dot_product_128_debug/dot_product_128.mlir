func.func @dot_product(%arg0: tensor<128xi32> {secret.secret}, %arg1: tensor<128xi32> {secret.secret}) -> i32 {
  %c0 = arith.constant 0 : index
  %c0_si32 = arith.constant 0 : i32
  %0 = affine.for %arg2 = 0 to 128 iter_args(%iter = %c0_si32) -> (i32) {
    %1 = tensor.extract %arg0[%arg2] : tensor<128xi32>
    %2 = tensor.extract %arg1[%arg2] : tensor<128xi32>
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %iter, %3 : i32
    affine.yield %4 : i32
  }
  return %0 : i32
}
