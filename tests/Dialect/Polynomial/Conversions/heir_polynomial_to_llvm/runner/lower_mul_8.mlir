#ideal_8 = #polynomial.int_polynomial<1 + x**3>
!coeff_ty_8 = !mod_arith.int<7:i32>
#ring_8 = #polynomial.ring<coefficientType=!coeff_ty_8, polynomialModulus=#ideal_8>
!poly_ty_8 = !polynomial.polynomial<ring=#ring_8>

func.func public @test_8() -> memref<3xi32> {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<-4 + x**1> : !poly_ty_8
  %1 = polynomial.constant int<-1 + 3x**1> : !poly_ty_8
  %2 = polynomial.mul %0, %1 : !poly_ty_8

  %3 = polynomial.to_tensor %2 : !poly_ty_8 -> tensor<3x!coeff_ty_8>
  %tensor = mod_arith.extract %3 : tensor<3x!coeff_ty_8> -> tensor<3xi32>


  %ref = bufferization.to_buffer %tensor : tensor<3xi32> to memref<3xi32>
  return %ref : memref<3xi32>
}
