#ideal_7 = #polynomial.int_polynomial<3 + 5 x**12>
!coeff_ty_7 = !mod_arith.int<16:i32>
#ring_7 = #polynomial.ring<coefficientType=!coeff_ty_7, polynomialModulus=#ideal_7>
!poly_ty_7 = !polynomial.polynomial<ring=#ring_7>

func.func public @test_7() -> memref<12xi32> {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_7
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_7
  %2 = polynomial.mul %0, %1 : !poly_ty_7

  %3 = polynomial.to_tensor %2 : !poly_ty_7 -> tensor<12x!coeff_ty_7>
  %tensor = mod_arith.extract %3 : tensor<12x!coeff_ty_7> -> tensor<12xi32>


  %ref = bufferization.to_buffer %tensor : tensor<12xi32> to memref<12xi32>
  return %ref : memref<12xi32>
}
