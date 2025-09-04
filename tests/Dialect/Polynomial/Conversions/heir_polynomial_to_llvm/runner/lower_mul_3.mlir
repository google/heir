#ideal_3 = #polynomial.int_polynomial<1 + x**12>
!coeff_ty_3 = !mod_arith.int<15:i32>
#ring_3 = #polynomial.ring<coefficientType=!coeff_ty_3, polynomialModulus=#ideal_3>
!poly_ty_3 = !polynomial.polynomial<ring=#ring_3>

func.func public @test_3() -> memref<12xi32> {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_3
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_3
  %2 = polynomial.mul %0, %1 : !poly_ty_3

  %3 = polynomial.to_tensor %2 : !poly_ty_3 -> tensor<12x!coeff_ty_3>
  %tensor = mod_arith.extract %3 : tensor<12x!coeff_ty_3> -> tensor<12xi32>


  %ref = bufferization.to_buffer %tensor : tensor<12xi32> to memref<12xi32>
  return %ref : memref<12xi32>
}
