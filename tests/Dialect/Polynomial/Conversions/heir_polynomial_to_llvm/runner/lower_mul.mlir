#ideal = #polynomial.int_polynomial<1 + x**12>
!coeff_ty = !mod_arith.int<65536:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#ideal>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func public @test_mul() -> !poly_ty {
  // 1 - x^9 + x^10 + x^11
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty
  %1 = polynomial.constant int<1 + x**11> : !poly_ty
  %2 = polynomial.mul %0, %1 : !poly_ty
  return %2 : !poly_ty
}
