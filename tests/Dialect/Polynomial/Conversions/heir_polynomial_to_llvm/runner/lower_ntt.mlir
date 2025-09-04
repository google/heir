// This follows from example 3.8 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

#cycl = #polynomial.int_polynomial<1 + x**4>
!coeff_ty = !mod_arith.int<7681:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl>
#root = #polynomial.primitive_root<value=1925:i32, degree=8:i32>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func public @test_ntt() -> tensor<4x!coeff_ty, #ring> {
  %coeffsRaw = arith.constant dense<[1,2,3,4]> : tensor<4xi32>
  %coeffs = mod_arith.encapsulate %coeffsRaw : tensor<4xi32> -> tensor<4x!coeff_ty>
  %poly = polynomial.from_tensor %coeffs : tensor<4x!coeff_ty> -> !poly_ty
  %res = polynomial.ntt %poly {root=#root} : !poly_ty -> tensor<4x!coeff_ty, #ring>
  return %res : tensor<4x!coeff_ty, #ring>
}
