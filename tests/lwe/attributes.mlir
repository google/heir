// RUN: not heir-opt --split-input-file %s 2>&1 | FileCheck %s

// This simply tests for syntax.


#encoding0 = #lwe.unspecified_bit_field_encoding<
  cleartext_bitwidth=3>

// CHECK-LABEL: test_valid_unspecified_lwe_attribute
func.func @test_valid_unspecified_lwe_attribute() {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    // CHECK: unspecified_bit_field_encoding
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16, #encoding0>
  return
}

// -----

#encoding1 = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>

// CHECK-LABEL: test_valid_lwe_attribute
func.func @test_valid_lwe_attribute() {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    // CHECK: bit_field_encoding
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16, #encoding1>
  return
}

// -----

#encoding2 = #lwe.bit_field_encoding<
  cleartext_start=30,
  cleartext_bitwidth=3>

// expected-error@below {{cleartext starting bit index (30) is outside the legal range [0, 15]}}
func.func @test_invalid_lwe_attribute() {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16, #encoding2>
  return
}

// -----

#encoding3 = #lwe.unspecified_bit_field_encoding<
  cleartext_bitwidth=8>

// expected-error@below {{tensor element type's bitwidth 4 is too small to store the cleartext, which has bit width 8}}
func.func @test_invalid_unspecified_lwe_attribute() {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i4
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi4, #encoding3>
  return
}

// -----

#generator = #_polynomial.polynomial<1 + x**1024>
#ring = #_polynomial.ring<cmod=65536, ideal=#generator>
#coeff_encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
// CHECK-LABEL: test_valid_coefficient_encoding_attr
// CHECK: polynomial_coefficient_encoding
func.func @test_valid_coefficient_encoding_attr(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = _polynomial.from_tensor %coeffs1 : tensor<10xi16> -> !_polynomial.polynomial<#ring>
  %poly2 = _polynomial.from_tensor %coeffs2 : tensor<10xi16> -> !_polynomial.polynomial<#ring>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!_polynomial.polynomial<#ring>, #coeff_encoding>
  return
}

// -----

#coeff_encoding1 = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
// expected-error@below {{must have `_polynomial.polynomial` element type}}
func.func @test_invalid_coefficient_encoding_type(%a : i16, %b: i16) {
  %rlwe_ciphertext = tensor.from_elements %a, %b : tensor<2xi16, #coeff_encoding1>
  return
}

// -----

#generator2 = #_polynomial.polynomial<1 + x**1024>
#ring2 = #_polynomial.ring<cmod=65536, ideal=#generator2>
#coeff_encoding2 = #lwe.polynomial_coefficient_encoding<cleartext_start=30, cleartext_bitwidth=3>
// expected-error@below {{cleartext starting bit index (30) is outside the legal range [0, 15]}}
func.func @test_invalid_coefficient_encoding_width(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = _polynomial.from_tensor %coeffs1 : tensor<10xi16> -> !_polynomial.polynomial<#ring2>
  %poly2 = _polynomial.from_tensor %coeffs2 : tensor<10xi16> -> !_polynomial.polynomial<#ring2>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!_polynomial.polynomial<#ring2>, #coeff_encoding2>
  return
}

// -----

#eval_enc2 = #lwe.polynomial_evaluation_encoding<cleartext_start=14, cleartext_bitwidth=3>
// expected-error@below {{must have `_polynomial.polynomial` element type}}
func.func @test_invalid_evaluation_encoding_type() {
  %a = arith.constant dense<[2, 2, 5]> : tensor<3xi32, #eval_enc2>
  return
}

// -----

#inverse_canonical_enc2 = #lwe.inverse_canonical_embedding_encoding<cleartext_start=14, cleartext_bitwidth=4>
// expected-error@below {{must have `_polynomial.polynomial` element type}}
func.func @test_invalid_inverse_canonical_embedding_encoding() {
  %a = arith.constant dense<[2, 2, 5]> : tensor<3xi32, #inverse_canonical_enc2>
  return
}
