#ideal_6 = #polynomial.int_polynomial<3 + x**12>
!coeff_ty_6 = !mod_arith.int<16:i32>
#ring_6 = #polynomial.ring<coefficientType=!coeff_ty_6, polynomialModulus=#ideal_6>
!poly_ty_6 = !polynomial.polynomial<ring=#ring_6>

func.func public @test_6() -> !poly_ty_6 {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_6
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_6
  %2 = polynomial.mul %0, %1 : !poly_ty_6
  return %2 : !poly_ty_6
}
