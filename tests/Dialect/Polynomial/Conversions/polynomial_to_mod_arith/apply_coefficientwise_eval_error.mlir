// RUN: heir-opt --polynomial-to-mod-arith --verify-diagnostics %s

// Only bodies that select RNS limbs can be lowered in eval form.

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465 : i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>
!poly_ty = !polynomial.polynomial<ring=#ring, form=eval>

func.func @test_eval_form_not_lowered(%p0 : !poly_ty) -> !poly_ty {
  // expected-error@below {{failed to legalize operation 'polynomial.apply_coefficientwise'}}
  %0 = polynomial.apply_coefficientwise (%p0 : !poly_ty) {
  ^body(%coeff: !coeff_ty, %degree: index):
    %1 = mod_arith.add %coeff, %coeff : !coeff_ty
    polynomial.yield %1 : !coeff_ty
  } -> !poly_ty
  return %0 : !poly_ty
}
