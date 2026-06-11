// RUN: heir-opt --polynomial-to-mod-arith --verify-diagnostics --split-input-file %s

#poly = #polynomial.int_polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
!poly_ty = !polynomial.polynomial<ring=#ring, form=eval>

func.func @eval_constant() -> !poly_ty {
  // expected-error@+1 {{failed to legalize operation}}
  %0 = polynomial.constant int<1> : !poly_ty
  return %0 : !poly_ty
}

// -----

#poly = #polynomial.int_polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
!poly_ty = !polynomial.polynomial<ring=#ring, form=eval>

func.func @eval_nonconstant_monomial(%coeff: i32) -> !poly_ty {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{failed to legalize operation}}
  %0 = polynomial.monomial %coeff, %c1 : (i32, index) -> !poly_ty
  return %0 : !poly_ty
}
