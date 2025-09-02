#ideal_4 = #polynomial.int_polynomial<1 + x**12>
!coeff_ty_4 = !mod_arith.int<17:i32>
#ring_4 = #polynomial.ring<coefficientType=!coeff_ty_4, polynomialModulus=#ideal_4>
!poly_ty_4 = !polynomial.polynomial<ring=#ring_4>

func.func public @test_4() -> memref<12xi32> {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_4
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_4
  %2 = polynomial.mul %0, %1 : !poly_ty_4

  %3 = polynomial.to_tensor %2 : !poly_ty_4 -> tensor<12x!coeff_ty_4>
  %tensor = mod_arith.extract %3 : tensor<12x!coeff_ty_4> -> tensor<12xi32>


  %ref = bufferization.to_buffer %tensor : tensor<12xi32> to memref<12xi32>
  return %ref : memref<12xi32>
}
