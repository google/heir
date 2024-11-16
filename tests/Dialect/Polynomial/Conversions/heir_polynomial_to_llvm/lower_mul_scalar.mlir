// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

!coeff_ty = !mod_arith.int<4294967296:i64>
!poly_ty = !polynomial.polynomial<ring=<coefficientType=!coeff_ty, polynomialModulus=#polynomial.int_polynomial<-1 + x**2047>>>

// CHECK-LABEL: func.func @test_lower_mul_scalar
// CHECK-SAME: (%[[ARG:[^:]*]]: [[T:.*]])
func.func @test_lower_mul_scalar(%arg0: !poly_ty) -> !poly_ty {
  // CHECK: %[[C2:.*]] = mod_arith.constant 2
  %c2 = mod_arith.constant 2 : !coeff_ty
  // CHECK: %[[C2EXT:.*]] = mod_arith.extract %[[C2]]
  // CHECK: %[[SPLAT:.*]] = tensor.splat %[[C2EXT]]
  // CHECK: %[[SPLAT_ENC:.*]] = mod_arith.encapsulate %[[SPLAT]]
  // CHECK: mod_arith.mul %[[ARG]], %[[SPLAT_ENC]] : [[T]]
  %8 = polynomial.mul_scalar %arg0, %c2 : !poly_ty, !coeff_ty
  return %8 : !poly_ty
}
