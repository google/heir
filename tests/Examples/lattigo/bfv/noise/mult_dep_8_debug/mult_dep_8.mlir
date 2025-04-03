func.func @mult_dep(
    %arg0: i16 {secret.secret}
    ) -> i16 {
    %0 = arith.muli %arg0, %arg0 : i16
    %1 = arith.muli %0, %arg0 : i16
    %2 = arith.muli %1, %arg0 : i16
    %3 = arith.muli %2, %arg0 : i16
    %4 = arith.muli %3, %arg0 : i16
    %5 = arith.muli %4, %arg0 : i16
    %6 = arith.muli %5, %arg0 : i16
    return %6 : i16
}
