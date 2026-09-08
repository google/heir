// RUN: heir-opt --polynomial-coeff-folding %s | FileCheck %s

!poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>

module {
  func.func @doctest(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = arith.addf %arg0, %cst : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <3 + x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %0 : f32
    return %1 : f32
  }
}
