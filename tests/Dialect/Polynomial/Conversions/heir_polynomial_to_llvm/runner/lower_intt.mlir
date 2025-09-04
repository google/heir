// This follows from example 3.10 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

#cycl = #polynomial.int_polynomial<1 + x**4>
!coeff_ty = !mod_arith.int<7681:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl>
#root = #polynomial.primitive_root<value=1925:i32, degree=8:i32>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func public @test_intt() -> !poly_ty {
  %coeffsRaw = arith.constant dense<[1467,2807,3471,7621]> : tensor<4xi32>
  %coeffs = tensor.cast %coeffsRaw : tensor<4xi32> to tensor <4xi32, #ring>
  %coeffs_enc = mod_arith.encapsulate %coeffs : tensor<4xi32, #ring> -> tensor<4x!coeff_ty, #ring>
  %0 = polynomial.intt %coeffs_enc {root=#root} : tensor<4x!coeff_ty, #ring> -> !poly_ty
  return %0 : !poly_ty
}
