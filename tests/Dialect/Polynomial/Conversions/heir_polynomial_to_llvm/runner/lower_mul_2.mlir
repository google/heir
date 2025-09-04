#ideal_2 = #polynomial.int_polynomial<1 + x**12>
!coeff_ty_2 = !mod_arith.int<16:i32>
#ring_2 = #polynomial.ring<coefficientType=!coeff_ty_2, polynomialModulus=#ideal_2>
!poly_ty_2 = !polynomial.polynomial<ring=#ring_2>

func.func public @test_2() -> !poly_ty_2 {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_2
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_2
  %2 = polynomial.mul %0, %1 : !poly_ty_2
  return %2 : !poly_ty_2
}
