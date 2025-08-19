func.func @box_blur(%arg0: tensor<256xi16> {secret.secret}) -> tensor<256xi16> {
  %c256 = arith.constant 256 : index
  %c16 = arith.constant 16 : index
  %0 = affine.for %x = 0 to 16 iter_args(%arg0_x = %arg0) -> (tensor<256xi16>) {
    %1 = affine.for %y = 0 to 16 iter_args(%arg0_y = %arg0_x) -> (tensor<256xi16>) {
      %c0_si16 = arith.constant 0 : i16
      %2 = affine.for %j = -1 to 2 iter_args(%value_j = %c0_si16) -> (i16) {
        %6 = affine.for %i = -1 to 2 iter_args(%value_i = %value_j) -> (i16) {
          %7 = arith.addi %x, %i : index
          %8 = arith.muli %7, %c16 : index
          %9 = arith.addi %y, %j : index
          %10 = arith.addi %8, %9 : index
          %11 = arith.remui %10, %c256 : index
          %12 = tensor.extract %arg0[%11] : tensor<256xi16>
          %13 = arith.addi %value_i, %12 : i16
          affine.yield %13 : i16
        }
        affine.yield %6 : i16
      }
      %3 = arith.muli %c16, %x : index
      %4 = arith.addi %3, %y : index
      %5 = arith.remui %4, %c256 : index
      %6 = tensor.insert %2 into %arg0_y[%5] : tensor<256xi16>
      affine.yield %6 : tensor<256xi16>
    }
    affine.yield %1 : tensor<256xi16>
  }
  return %0 : tensor<256xi16>
}
