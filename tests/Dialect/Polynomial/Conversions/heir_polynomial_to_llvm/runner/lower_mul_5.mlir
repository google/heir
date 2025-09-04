#ideal_5 = #polynomial.int_polynomial<1 + x**12>
!coeff_ty_5 = !mod_arith.int<16:i32>
#ring_5 = #polynomial.ring<coefficientType=!coeff_ty_5, polynomialModulus=#ideal_5>
!poly_ty_5 = !polynomial.polynomial<ring=#ring_5>

func.func public @test_5() -> !poly_ty_5 {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**2> : !poly_ty_5
  %1 = polynomial.constant int<1 + x**3> : !poly_ty_5
  %2 = polynomial.mul %0, %1 : !poly_ty_5
  return %2 : !poly_ty_5
}
