#ideal = #polynomial.int_polynomial<1 + x**12>
!coeff_ty = !mod_arith.int<2147483647:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#ideal>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func public @test_add() -> memref<12xi32> {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty
  %1 = polynomial.constant int<1 + x**11> : !poly_ty
  %2 = polynomial.add %0, %1 : !poly_ty

  %3 = polynomial.to_tensor %2 : !poly_ty -> tensor<12x!coeff_ty>
  %4 = mod_arith.extract %3 : tensor<12x!coeff_ty> -> tensor<12xi32>
  %5 = bufferization.to_buffer %4 : tensor<12xi32> to memref<12xi32>
  return %5 : memref<12xi32>
}
