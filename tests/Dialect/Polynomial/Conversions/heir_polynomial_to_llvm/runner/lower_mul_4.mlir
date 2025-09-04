#ideal_4 = #polynomial.int_polynomial<1 + x**12>
!coeff_ty_4 = !mod_arith.int<17:i32>
#ring_4 = #polynomial.ring<coefficientType=!coeff_ty_4, polynomialModulus=#ideal_4>
!poly_ty_4 = !polynomial.polynomial<ring=#ring_4>

func.func public @test_4() -> !poly_ty_4 {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_4
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_4
  %2 = polynomial.mul %0, %1 : !poly_ty_4
  return %2 : !poly_ty_4
}
