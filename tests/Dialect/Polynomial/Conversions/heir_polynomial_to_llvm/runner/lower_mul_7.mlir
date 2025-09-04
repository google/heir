#ideal_7 = #polynomial.int_polynomial<3 + 5 x**12>
!coeff_ty_7 = !mod_arith.int<16:i32>
#ring_7 = #polynomial.ring<coefficientType=!coeff_ty_7, polynomialModulus=#ideal_7>
!poly_ty_7 = !polynomial.polynomial<ring=#ring_7>

func.func public @test_7() -> !poly_ty_7 {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_7
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_7
  %2 = polynomial.mul %0, %1 : !poly_ty_7
  return %2 : !poly_ty_7
}
