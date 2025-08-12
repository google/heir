func.func @roberts_cross(%img: tensor<256xi16> {secret.secret}) -> tensor<256xi16> {
  %c256 = arith.constant 256 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c-1 =  arith.constant -1 : index

  // Each point p = img[x][y], where x is row and y is column, in the new image will equal:
  // (img[x-1][y-1] - img[x][y])^2 + (img[x-1][y] - img[x][y-1])^2
  %r = affine.for %x = 0 to 16 iter_args(%imgx = %img) -> tensor<256xi16> {
    %1 = affine.for %y = 0 to 16 iter_args(%imgy = %imgx) -> tensor<256xi16> {

      // fetch img[x-1][y-1]
      %4 = arith.addi %x, %c-1 : index
      %5 = arith.muli %4, %c16 : index
      %6 = arith.addi %y, %c-1 : index
      %7 = arith.addi %5, %6 : index
      %8 = arith.remui %7, %c256 : index
      %9 = tensor.extract %img[%8] : tensor<256xi16>

      // fetch img[x][y]
      %10 = arith.muli %x, %c16 : index
      %11 = arith.addi %10, %y : index
      %12 = arith.remui %11, %c256 : index
      %13 = tensor.extract %img[%12] : tensor<256xi16>

      // subtract those two
      %14 = arith.subi %9, %13 : i16

      // fetch img[x-1][y]
      %15 = arith.addi %x, %c-1 : index
      %16 = arith.muli %15, %c16 : index
      %18 = arith.addi %16, %y : index
      %19 = arith.remui %18, %c256 : index
      %20 = tensor.extract %img[%19] : tensor<256xi16>

      // fetch img[x][y-1]
      %21 = arith.muli %x, %c16 : index
      %22 = arith.addi %y, %c-1 : index
      %23 = arith.addi %21, %22 : index
      %24 = arith.remui %23, %c256 : index
      %25 = tensor.extract %img[%24] : tensor<256xi16>

      // subtract those two
      %26 = arith.subi %20, %25 : i16

      // square each difference
      %27 = arith.muli %14, %14 :  i16
      %28 = arith.muli %26, %26 :  i16

      // add the squares
      %29 = arith.addi %27, %28 : i16

      // save to result[x][y]
      %30 = tensor.insert %29 into %imgy[%12] : tensor<256xi16>
      affine.yield %30: tensor<256xi16>
    }
    affine.yield %1 : tensor<256xi16>
  }
  return %r : tensor<256xi16>
}
