// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

#ring = #polynomial.ring<coefficientType = !mod_arith.int<1095233372161 : i64>, polynomialModulus = <1 + x**8>>
!poly_ty = !polynomial.polynomial<ring=#ring, form=coeff>
!coeff_ty = !mod_arith.int<1095233372161 : i64>

module {
  // CHECK: func.func @test_apply_coefficientwise_forces_coeff
  // CHECK-SAME: (%[[ARG0:.*]]: ![[POLY_EVAL:.*]]) -> ![[POLY_COEFF:.*]] {
  func.func @test_apply_coefficientwise_forces_coeff(%p0 : !poly_ty) -> !poly_ty {
    // CHECK: %[[MUL:.*]] = polynomial.mul %[[ARG0]], %[[ARG0]] : ![[POLY_EVAL]]
    %1 = polynomial.mul %p0, %p0 : !poly_ty

    // Since apply_coefficientwise is now marked as COEFF form,
    // and its input %1 is currently in EVAL form (from the mul),
    // the solver should force an INTT here.

    // CHECK: %[[INTT:.*]] = polynomial.intt %[[MUL]] : ![[POLY_EVAL]]
    // CHECK: %[[APPLY:.*]] = polynomial.apply_coefficientwise(%[[INTT]] : ![[POLY_COEFF]])
    %2 = polynomial.apply_coefficientwise (%1 : !poly_ty) {
    ^body(%coeff: !coeff_ty, %degree: index):
      %3 = mod_arith.add %coeff, %coeff : !coeff_ty
      polynomial.yield %3 : !coeff_ty
    } -> !poly_ty

    // CHECK: return %[[APPLY]] : ![[POLY_COEFF]]
    return %2 : !poly_ty
  }
}
