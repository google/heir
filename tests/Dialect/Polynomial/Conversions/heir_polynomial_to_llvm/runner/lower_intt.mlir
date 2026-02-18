// This follows from example 3.10 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

#cycl = #polynomial.int_polynomial<1 + x**4>
!coeff_ty = !mod_arith.int<7681:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl>
#root_val = #mod_arith.value<1925:!coeff_ty>
#root = #polynomial.primitive_root<value=#root_val, degree=8:i32>
!poly_ty = !polynomial.polynomial<ring=#ring>
!ntt_poly_ty = !polynomial.polynomial<ring=#ring, form=eval>

func.func public @test_intt() -> !poly_ty {
  %coeffs = arith.constant dense<[1467,2807,3471,7621]> : tensor<4xi32>
  %coeffs_enc = mod_arith.encapsulate %coeffs : tensor<4xi32> -> tensor<4x!coeff_ty>
  %poly = polynomial.from_tensor %coeffs_enc : tensor<4x!coeff_ty> -> !ntt_poly_ty
  %0 = polynomial.intt %poly {root=#root} : !ntt_poly_ty
  return %0 : !poly_ty
}
