#ideal = #polynomial.int_polynomial<1 + x**12>
!coeff_ty = !mod_arith.int<65536:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#ideal>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func public @test_leading_term() -> memref<2xi32> {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + 2x**10> : !poly_ty
  %2, %1 = polynomial.leading_term %0 : !poly_ty -> (index, !coeff_ty)
  %3 = mod_arith.extract %1 : !coeff_ty -> i32

  %4 = memref.alloca() : memref<2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  memref.store %3, %4[%c0] : memref<2xi32>
  %5 = arith.index_cast %2 : index to i32
  memref.store %5, %4[%c1] : memref<2xi32>
  return %4 : memref<2xi32>
}
