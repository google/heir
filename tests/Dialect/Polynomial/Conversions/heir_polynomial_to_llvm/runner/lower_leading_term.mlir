#ideal = #polynomial.int_polynomial<1 + x**12>
!coeff_ty = !mod_arith.int<65536:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#ideal>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func public @test_leading_term() -> (index, !coeff_ty) {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + 2x**10> : !poly_ty
  %1:2 = polynomial.leading_term %0 : !poly_ty -> (index, !coeff_ty)
  return %1#0, %1#1 : index, !coeff_ty
}
