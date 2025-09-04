#ideal_1 = #polynomial.int_polynomial<1 + x**12>
!coeff_ty_1 = !mod_arith.int<2147483647:i32>
#ring_1 = #polynomial.ring<coefficientType=!coeff_ty_1, polynomialModulus=#ideal_1>
!poly_ty_1 = !polynomial.polynomial<ring=#ring_1>

func.func public @test_1() -> !poly_ty_1 {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_1
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_1
  %2 = polynomial.mul %0, %1 : !poly_ty_1
  return %2 : !poly_ty_1
}
