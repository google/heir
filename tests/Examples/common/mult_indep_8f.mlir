func.func @mult_indep(
    %arg0: f32 {secret.secret},
    %arg1: f32 {secret.secret},
    %arg2: f32 {secret.secret},
    %arg3: f32 {secret.secret},
    %arg4: f32 {secret.secret},
    %arg5: f32 {secret.secret},
    %arg6: f32 {secret.secret},
    %arg7: f32 {secret.secret}
    ) -> f32 {
    %0 = arith.mulf %arg0, %arg1 : f32
    %1 = arith.mulf %0, %arg2 : f32
    %2 = arith.mulf %1, %arg3 : f32
    %3 = arith.mulf %2, %arg4 : f32
    %4 = arith.mulf %3, %arg5 : f32
    %5 = arith.mulf %4, %arg6 : f32
    %6 = arith.mulf %5, %arg7 : f32
    return %6 : f32
}
