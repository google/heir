func.func @mult_dep(
    %arg0: f32 {secret.secret}
    ) -> f32 {
    %0 = arith.mulf %arg0, %arg0 : f32
    %1 = arith.mulf %0, %arg0 : f32
    %2 = arith.mulf %1, %arg0 : f32
    %3 = arith.mulf %2, %arg0 : f32
    %4 = arith.mulf %3, %arg0 : f32
    %5 = arith.mulf %4, %arg0 : f32
    %6 = arith.mulf %5, %arg0 : f32
    return %6 : f32
}
