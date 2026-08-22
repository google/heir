// RUN: heir-opt --polynomial-coeff-folding %s | FileCheck %s

!poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>

module {
  // CHECK: func @e2e_test
  func.func @e2e_test(%arg0: f32) -> f32 {
    %cst1 = arith.constant 2.000000e+00 : f32
    %cst2 = arith.constant 3.000000e+00 : f32

    // Before eval
    %0 = arith.addf %arg0, %cst1 : f32

    // Eval: P(x) = 1 + x
    // P(x + 2) = 3 + x
    %1 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %0 : f32

    // After eval: (3 + x) + 3 = 6 + x
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <6 + x> : ![[POLY:.*]]>, %arg0 : f32
    %2 = arith.addf %1, %cst2 : f32

    return %2 : f32
  }
}
