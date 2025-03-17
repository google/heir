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
    %7 = arith.muli %6, %arg0 : i16
    %8 = arith.muli %7, %arg0 : i16
    %9 = arith.muli %8, %arg0 : i16
    %10 = arith.muli %9, %arg0 : i16
    %11 = arith.muli %10, %arg0 : i16
    %12 = arith.muli %11, %arg0 : i16
    %13 = arith.muli %12, %arg0 : i16
    %14 = arith.muli %13, %arg0 : i16
    return %6 : i16
}
