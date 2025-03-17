func.func @mult_indep(
    %arg0: i16 {secret.secret},
    %arg1: i16 {secret.secret},
    %arg2: i16 {secret.secret},
    %arg3: i16 {secret.secret},
    %arg4: i16 {secret.secret},
    %arg5: i16 {secret.secret},
    %arg6: i16 {secret.secret},
    %arg7: i16 {secret.secret},
    %arg8: i16 {secret.secret},
    %arg9: i16 {secret.secret},
    %arg10: i16 {secret.secret},
    %arg11: i16 {secret.secret},
    %arg12: i16 {secret.secret},
    %arg13: i16 {secret.secret},
    %arg14: i16 {secret.secret},
    %arg15: i16 {secret.secret},
    %arg16: i16 {secret.secret},
    %arg17: i16 {secret.secret},
    %arg18: i16 {secret.secret},
    %arg19: i16 {secret.secret},
    %arg20: i16 {secret.secret},
    %arg21: i16 {secret.secret},
    %arg22: i16 {secret.secret},
    %arg23: i16 {secret.secret},
    %arg24: i16 {secret.secret},
    %arg25: i16 {secret.secret},
    %arg26: i16 {secret.secret},
    %arg27: i16 {secret.secret},
    %arg28: i16 {secret.secret},
    %arg29: i16 {secret.secret},
    %arg30: i16 {secret.secret},
    %arg31: i16 {secret.secret}
    ) -> i16 {
    %0 = arith.muli %arg0, %arg1 : i16
    %1 = arith.muli %0, %arg2 : i16
    %2 = arith.muli %1, %arg3 : i16
    %3 = arith.muli %2, %arg4 : i16
    %4 = arith.muli %3, %arg5 : i16
    %5 = arith.muli %4, %arg6 : i16
    %6 = arith.muli %5, %arg7 : i16
    %7 = arith.muli %6, %arg8 : i16
    %8 = arith.muli %7, %arg9 : i16
    %9 = arith.muli %8, %arg10 : i16
    %10 = arith.muli %9, %arg11 : i16
    %11 = arith.muli %10, %arg12 : i16
    %12 = arith.muli %11, %arg13 : i16
    %13 = arith.muli %12, %arg14 : i16
    %14 = arith.muli %13, %arg15 : i16
    %15 = arith.muli %14, %arg16 : i16
    %16 = arith.muli %15, %arg17 : i16
    %17 = arith.muli %16, %arg18 : i16
    %18 = arith.muli %17, %arg19 : i16
    %19 = arith.muli %18, %arg20 : i16
    %20 = arith.muli %19, %arg21 : i16
    %21 = arith.muli %20, %arg22 : i16
    %22 = arith.muli %21, %arg23 : i16
    %23 = arith.muli %22, %arg24 : i16
    %24 = arith.muli %23, %arg25 : i16
    %25 = arith.muli %24, %arg26 : i16
    %26 = arith.muli %25, %arg27 : i16
    %27 = arith.muli %26, %arg28 : i16
    %28 = arith.muli %27, %arg29 : i16
    %29 = arith.muli %28, %arg30 : i16
    %30 = arith.muli %29, %arg31 : i16
    return %30 : i16
}
