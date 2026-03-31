#poly_mod = #polynomial.int_polynomial<1 + x**8>

!coeff17 = !mod_arith.int<17 : i32>
#ring17 = #polynomial.ring<coefficientType = !coeff17, polynomialModulus = #poly_mod>
!poly17 = !polynomial.polynomial<ring = #ring17, form=coeff>
!poly17e = !polynomial.polynomial<ring = #ring17, form=eval>
!tensor17 = tensor<8x!coeff17>

func.func public @test_mul_ntt_roundtrip_modarith() -> !poly17 {
  %coeffs_raw = arith.constant dense<[1, 2, 3, 8, 7, 6, 3, 4]> : tensor<8xi32>
  %coeffs = mod_arith.encapsulate %coeffs_raw : tensor<8xi32> -> !tensor17
  // Creates a polynomial in coeff form
  %poly = polynomial.from_tensor %coeffs : !tensor17 -> !poly17
  %ntt_poly = polynomial.ntt %poly : !poly17
  %intt_poly = polynomial.intt %ntt_poly : !poly17e
  return %intt_poly : !poly17
}
