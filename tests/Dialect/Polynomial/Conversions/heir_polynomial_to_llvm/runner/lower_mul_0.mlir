#ideal_0 = #polynomial.int_polynomial<1 + x**12>
!coeff_ty_0 = !mod_arith.int<4294967296:i64>
#ring_0 = #polynomial.ring<coefficientType=!coeff_ty_0, polynomialModulus=#ideal_0>
!poly_ty_0 = !polynomial.polynomial<ring=#ring_0>

func.func public @test_0() -> !poly_ty_0 {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_0
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_0
  %2 = polynomial.mul %0, %1 : !poly_ty_0
  return %2 : !poly_ty_0
}
