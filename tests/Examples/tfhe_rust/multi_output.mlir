module {
  func.func @multi_output(%arg0: i8 {secret.secret}) -> (i8, i8) {
    %c2_i8 = arith.constant 2 : i8
    %c3_i8 = arith.constant 3 : i8
    %0 = arith.addi %arg0, %c3_i8 : i8
    %1 = arith.muli %arg0, %c2_i8 : i8
    return %0, %1 : i8, i8
  }
}
