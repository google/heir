#ideal_9 = #polynomial.int_polynomial<1 + x**3>
!coeff_ty_9 = !mod_arith.int<8:i32>
#ring_9 = #polynomial.ring<coefficientType=!coeff_ty_9, polynomialModulus=#ideal_9>
!poly_ty_9 = !polynomial.polynomial<ring=#ring_9>

func.func public @test_9() -> !poly_ty_9 {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<-4 + x**1> : !poly_ty_9
  %1 = polynomial.constant int<-1 + 3x**1> : !poly_ty_9
  %2 = polynomial.mul %0, %1 : !poly_ty_9
  return %2 : !poly_ty_9
}
