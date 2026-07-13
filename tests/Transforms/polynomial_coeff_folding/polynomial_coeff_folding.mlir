// RUN: heir-opt --polynomial-coeff-folding %s | FileCheck %s

!poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>

module {
  // --- Before Eval ---

  // CHECK: func @before_addf
  func.func @before_addf(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = arith.addf %arg0, %cst : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <3 + x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %0 : f32
    return %1 : f32
  }

  // CHECK: func @before_mulf
  func.func @before_mulf(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = arith.mulf %arg0, %cst : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <1 + 2x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %0 : f32
    return %1 : f32
  }

  // CHECK: func @before_subf
  func.func @before_subf(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = arith.subf %arg0, %cst : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <-1 + x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %0 : f32
    return %1 : f32
  }

  // CHECK: func @before_divf
  func.func @before_divf(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = arith.divf %arg0, %cst : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <1 + 0.5x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %0 : f32
    return %1 : f32
  }

  // --- After Eval ---

  // CHECK: func @after_addf
  func.func @after_addf(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %arg0 : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <3 + x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = arith.addf %0, %cst : f32
    return %1 : f32
  }

  // CHECK: func @after_mulf
  func.func @after_mulf(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %arg0 : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <2 + 2x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = arith.mulf %0, %cst : f32
    return %1 : f32
  }

  // CHECK: func @after_subf
  func.func @after_subf(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %arg0 : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <-1 + x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = arith.subf %0, %cst : f32
    return %1 : f32
  }

  // CHECK: func @after_divf
  func.func @after_divf(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = polynomial.eval #polynomial.typed_float_polynomial<1.000000e+00 + 1.000000e+00 x> : !poly_ty, %arg0 : f32
    // CHECK: polynomial.eval #polynomial<typed_float_polynomial <0.5 + 0.5x> : ![[POLY:.*]]>, %arg0 : f32
    %1 = arith.divf %0, %cst : f32
    return %1 : f32
  }
}
