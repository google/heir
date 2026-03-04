// RUN: heir-opt --verify-diagnostics --split-input-file %s

#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#poly_1024 = #polynomial.int_polynomial<1 + x**1024>
!plaintext_coefficient_modulus = !mod_arith.int<65537 : i64>
#plaintext_ring = #polynomial.ring<
  coefficientType=!plaintext_coefficient_modulus,
  polynomialModulus=#poly_1024>
#plaintext_space = #lwe.plaintext_space<
  ring=#plaintext_ring,
  encoding=#encoding>

!pt = !lwe.lwe_plaintext<
  plaintext_space=#plaintext_space>

func.func @test_rlwe_encode_elementwise(%arg0: tensor<4x1024xi16>) -> tensor<4x!pt> {
  %0 = lwe.rlwe_encode %arg0 {encoding = #encoding, ring = #plaintext_ring} : tensor<4x1024xi16> -> tensor<4x!pt>
  return %0 : tensor<4x!pt>
}

// -----

#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#poly_1024 = #polynomial.int_polynomial<1 + x**1024>
!plaintext_coefficient_modulus = !mod_arith.int<65537 : i64>
#plaintext_ring = #polynomial.ring<
  coefficientType=!plaintext_coefficient_modulus,
  polynomialModulus=#poly_1024>
#plaintext_space = #lwe.plaintext_space<
  ring=#plaintext_ring,
  encoding=#encoding>

!pt = !lwe.lwe_plaintext<
  plaintext_space=#plaintext_space>

func.func @test_rlwe_encode_elementwise_error(%arg0: tensor<4x1024xi16>) -> tensor<5x!pt> {
  // expected-error@+1 {{expected all tensor results to have the same shape as mappable operands, but found shape (4) at operand 0 and shape (5) at result 0}}
  %0 = lwe.rlwe_encode %arg0 {encoding = #encoding, ring = #plaintext_ring} : tensor<4x1024xi16> -> tensor<5x!pt>
  return %0 : tensor<5x!pt>
}
