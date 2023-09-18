// RUN: not heir-opt --split-input-file %s 2>&1 | FileCheck %s

// This simply tests for syntax.

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

#generator = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=65536, ideal=#generator>
#coeff_encoding = #lwe.poly_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
// CHECK-LABEL: test_valid_coefficient_encoding_attr
// CHECK: poly_coefficient_encoding
func.func @test_valid_coefficient_encoding_attr(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = poly.from_tensor %coeffs1 : tensor<10xi16> -> !poly.poly<#ring>
  %poly2 = poly.from_tensor %coeffs2 : tensor<10xi16> -> !poly.poly<#ring>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring>, #coeff_encoding>
  return
}

// -----

#coeff_encoding1 = #lwe.poly_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
// expected-error@below {{must have `poly.poly` element type}}
func.func @test_invalid_coefficient_encoding_type(%a : i16, %b: i16) {
  %rlwe_ciphertext = tensor.from_elements %a, %b : tensor<2xi16, #coeff_encoding1>
  return
}

// -----

#generator2 = #poly.polynomial<1 + x**1024>
#ring2 = #poly.ring<cmod=65536, ideal=#generator2>
#coeff_encoding2 = #lwe.poly_coefficient_encoding<cleartext_start=30, cleartext_bitwidth=3>
// expected-error@below {{cleartext starting bit index (30) is outside the legal range [0, 15]}}
func.func @test_invalid_coefficient_encoding_width(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = poly.from_tensor %coeffs1 : tensor<10xi16> -> !poly.poly<#ring2>
  %poly2 = poly.from_tensor %coeffs2 : tensor<10xi16> -> !poly.poly<#ring2>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring2>, #coeff_encoding2>
  return
}

// -----

#generator3 = #poly.polynomial<1 + x**1024>
#ring3 = #poly.ring<cmod=65536, ideal=#generator3>
// CHECK-LABEL: test_valid_evaluation_encoding
// CHECK: poly_evaluation_encoding
#eval_enc = #lwe.poly_evaluation_encoding<cleartext_start=14, cleartext_bitwidth=3>
func.func @test_valid_evaluation_encoding(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = poly.from_tensor %coeffs1 : tensor<10xi16> -> !poly.poly<#ring3, #eval_enc>
  %poly2 = poly.from_tensor %coeffs2 : tensor<10xi16> -> !poly.poly<#ring3, #eval_enc>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring3, #eval_enc>, #eval_enc>
  return
}

// -----

#eval_enc2 = #lwe.poly_evaluation_encoding<cleartext_start=14, cleartext_bitwidth=3>
// expected-error@below {{must have `poly.poly` element type}}
func.func @test_invalid_evaluation_encoding_type() {
  %a = arith.constant dense<[2, 2, 5]> : tensor<3xi32, #eval_enc2>
  return
}

// -----

#generator4 = #poly.polynomial<1 + x**1024>
#ring4 = #poly.ring<cmod=65536, ideal=#generator4>
// CHECK-LABEL: test_valid_rounded_embedding_encoding
// CHECK: poly_rounded_embedding_encoding
#rounded_enc = #lwe.poly_rounded_embedding_encoding<cleartext_start=14, cleartext_bitwidth=4>
func.func @test_valid_rounded_embedding_encoding(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = poly.from_tensor %coeffs1 : tensor<10xi16> -> !poly.poly<#ring4, #rounded_enc>
  %poly2 = poly.from_tensor %coeffs2 : tensor<10xi16> -> !poly.poly<#ring4, #rounded_enc>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring4, #rounded_enc>, #rounded_enc>
  return
}

// -----

#rounded_enc2 = #lwe.poly_rounded_embedding_encoding<cleartext_start=14, cleartext_bitwidth=4>
// expected-error@below {{must have `poly.poly` element type}}
func.func @test_invalid_rounded_embedding_encoding() {
  %a = arith.constant dense<[2, 2, 5]> : tensor<3xi32, #rounded_enc2>
  return
}
