func.func @main(%arg0: i16) -> i16 {
  %c1 = arith.constant 1: i16
  %c2 = arith.constant 2 : i16
  %c3 = arith.constant 3: i16

  // %3 = arith.muli %arg0, %c2 : i16
  %4 = arith.addi %c2, %c1 : i16
  %5 = arith.addi %4, %c3 : i16

  // %5 = arith.muli %4, %c3 : i16
  %6 = arith.addi %5, %c2 : i16

  return %6 : i16
}
