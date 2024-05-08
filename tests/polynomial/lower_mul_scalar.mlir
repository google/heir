// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

!poly_ty = !polynomial.polynomial<ring=<coefficientType = i64, coefficientModulus = 18446744073709551616 : i65, polynomialModulus=#polynomial.int_polynomial<-1 + x**2047>>>

// CHECK-LABEL: func.func @test_lower_mul_scalar
// CHECK-SAME: (%[[ARG:.*]]: [[T:.*]])
func.func @test_lower_mul_scalar(%arg0: !poly_ty) -> !poly_ty {
  // CHECK: %[[C2:.*]] = arith.constant 2 : i64
  %c2 = arith.constant 2 : i64
  // CHECK: %[[SPLAT:.*]] = tensor.splat %[[C2]] : [[T]]
  // CHECK: arith.muli %[[ARG]], %[[SPLAT]] : [[T]]
  %8 = polynomial.mul_scalar %arg0, %c2 : !poly_ty, i64
  return %8 : !poly_ty
}
