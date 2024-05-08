// RUN: heir-opt --verify-diagnostics --split-input-file %s

#encoding2 = #lwe.bit_field_encoding<
  cleartext_start=30,
  cleartext_bitwidth=3>

// expected-error@below {{cleartext starting bit index (30) is outside the legal range [0, 15]}}
func.func @test_invalid_lwe_attribute() -> tensor<2xi16, #encoding2> {
  %c0 = arith.constant 0 : index
  %two = arith.constant 2 : i16
  %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16, #encoding2>
  return %coeffs1 : tensor<2xi16, #encoding2>
}

// -----

#encoding3 = #lwe.unspecified_bit_field_encoding<
  cleartext_bitwidth=8>

// expected-error@below {{tensor element type's bitwidth 4 is too small to store the cleartext, which has bit width 8}}
func.func @test_invalid_unspecified_lwe_attribute() -> tensor<2xi4, #encoding3> {
  %c0 = arith.constant 0 : index
  %two = arith.constant 2 : i4
  %coeffs1 = tensor.from_elements %two, %two : tensor<2xi4, #encoding3>
  return %coeffs1 : tensor<2xi4, #encoding3>
}

// -----

#coeff_encoding1 = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
func.func @test_invalid_coefficient_encoding_type(%a : i16, %b: i16) {
  // expected-error@below {{must have `polynomial.polynomial` element type}}
  %rlwe_ciphertext = tensor.from_elements %a, %b : tensor<2xi16, #coeff_encoding1>
  return
}

// -----

#generator2 = #polynomial.int_polynomial<1 + x**1024>
#ring2 = #polynomial.ring<coefficientType = i16, coefficientModulus = 123 : i16, polynomialModulus=#generator2>
#coeff_encoding2 = #lwe.polynomial_coefficient_encoding<cleartext_start=30, cleartext_bitwidth=3>
func.func @test_invalid_coefficient_encoding_width(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = polynomial.from_tensor %coeffs1 : tensor<10xi16> -> !polynomial.polynomial<ring=#ring2>
  %poly2 = polynomial.from_tensor %coeffs2 : tensor<10xi16> -> !polynomial.polynomial<ring=#ring2>
  // expected-error@below {{cleartext starting bit index (30) is outside the legal range [0, 15]}}
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!polynomial.polynomial<ring=#ring2>, #coeff_encoding2>
  return
}

// -----

#eval_enc2 = #lwe.polynomial_evaluation_encoding<cleartext_start=14, cleartext_bitwidth=3>
func.func @test_invalid_evaluation_encoding_type() {
  // expected-error@below {{must have `polynomial.polynomial` element type}}
  %a = arith.constant dense<[2, 2, 5]> : tensor<3xi32, #eval_enc2>
  return
}

// -----

#inverse_canonical_enc2 = #lwe.inverse_canonical_embedding_encoding<cleartext_start=14, cleartext_bitwidth=4>
func.func @test_invalid_inverse_canonical_embedding_encoding() {
  // expected-error@below {{must have `polynomial.polynomial` element type}}
  %a = arith.constant dense<[2, 2, 5]> : tensor<3xi32, #inverse_canonical_enc2>
  return
}
