func.func @mult_indep(
    %arg0: i16 {secret.secret},
    %arg1: i16 {secret.secret},
    %arg2: i16 {secret.secret},
    %arg3: i16 {secret.secret},
    %arg4: i16 {secret.secret},
    %arg5: i16 {secret.secret},
    %arg6: i16 {secret.secret},
    %arg7: i16 {secret.secret}
    ) -> i16 {
    %0 = arith.muli %arg0, %arg1 : i16
    %1 = arith.muli %0, %arg2 : i16
    %2 = arith.muli %1, %arg3 : i16
    %3 = arith.muli %2, %arg4 : i16
    %4 = arith.muli %3, %arg5 : i16
    %5 = arith.muli %4, %arg6 : i16
    %6 = arith.muli %5, %arg7 : i16
    return %6 : i16
}
