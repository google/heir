// RUN: heir-opt --split-input-file %s | FileCheck %s

// This simply tests for syntax.

#encoding = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>

// CHECK-LABEL: test_valid_lwe_encode
func.func @test_valid_lwe_encode() {
    %0 = arith.constant 0 : i1
    // CHECK: bit_field_encoding
    %2 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
  return
}

// -----

#unspecified_encoding = #lwe.unspecified_bit_field_encoding<
  cleartext_bitwidth=3>
!plaintext_unspecified = !lwe.lwe_plaintext<encoding = #unspecified_encoding>

// CHECK-LABEL: test_valid_lwe_unspecified_encode
func.func @test_valid_lwe_unspecified_encode() {
    %0 = arith.constant 0 : i1
    // CHECK: unspecified_bit_field_encoding
    %2 = lwe.encode %0 { encoding = #unspecified_encoding }: i1 to !plaintext_unspecified
  return
}

// -----

#encoding0 = #lwe.unspecified_bit_field_encoding<
  cleartext_bitwidth=3>

// CHECK-LABEL: test_valid_unspecified_lwe_attribute
func.func @test_valid_unspecified_lwe_attribute() -> tensor<2xi16, #encoding0> {
  %c0 = arith.constant 0 : index
  %two = arith.constant 2 : i16
  // CHECK: unspecified_bit_field_encoding
  %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16, #encoding0>
  return %coeffs1 : tensor<2xi16, #encoding0>
}

// -----

#encoding1 = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>

// CHECK-LABEL: test_valid_lwe_attribute
func.func @test_valid_lwe_attribute() -> tensor<2xi16, #encoding1> {
  %c0 = arith.constant 0 : index
  %two = arith.constant 2 : i16
  // CHECK: bit_field_encoding
  %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16, #encoding1>
  return %coeffs1 : tensor<2xi16, #encoding1>
}

// -----

#generator = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 65536 : i32, polynomialModulus=#generator>
#coeff_encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
// CHECK-LABEL: test_valid_coefficient_encoding_attr
// CHECK: polynomial_coefficient_encoding
func.func @test_valid_coefficient_encoding_attr(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = polynomial.from_tensor %coeffs1 : tensor<10xi16> -> !polynomial.polynomial<ring=#ring>
  %poly2 = polynomial.from_tensor %coeffs2 : tensor<10xi16> -> !polynomial.polynomial<ring=#ring>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!polynomial.polynomial<ring=#ring>, #coeff_encoding>
  return
}

// -----

#generator3 = #polynomial.int_polynomial<1 + x**1024>
#ring3 = #polynomial.ring<coefficientType = i32, coefficientModulus = 65536 : i32, polynomialModulus=#generator3>
// CHECK-LABEL: test_valid_evaluation_encoding
// CHECK: polynomial_evaluation_encoding
#eval_enc = #lwe.polynomial_evaluation_encoding<cleartext_start=14, cleartext_bitwidth=3>
func.func @test_valid_evaluation_encoding(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = polynomial.from_tensor %coeffs1 : tensor<10xi16> -> !polynomial.polynomial<ring=#ring3>
  %poly2 = polynomial.from_tensor %coeffs2 : tensor<10xi16> -> !polynomial.polynomial<ring=#ring3>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!polynomial.polynomial<ring=#ring3>, #eval_enc>
  return
}

// -----

#generator4 = #polynomial.int_polynomial<1 + x**1024>
#ring4 = #polynomial.ring<coefficientType = i32, coefficientModulus = 65536 : i32, polynomialModulus=#generator4>
// CHECK-LABEL: test_valid_inverse_canonical_embedding_encoding
// CHECK: inverse_canonical_embedding_encoding
#inverse_canonical_enc = #lwe.inverse_canonical_embedding_encoding<cleartext_start=14, cleartext_bitwidth=4>
func.func @test_valid_inverse_canonical_embedding_encoding(%coeffs1 : tensor<10xi16>, %coeffs2 : tensor<10xi16>) {
  %poly1 = polynomial.from_tensor %coeffs1 : tensor<10xi16> -> !polynomial.polynomial<ring=#ring4>
  %poly2 = polynomial.from_tensor %coeffs2 : tensor<10xi16> -> !polynomial.polynomial<ring=#ring4>
  %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!polynomial.polynomial<ring=#ring4>, #inverse_canonical_enc>
  return
}

// -----

#preserve_overflow = #lwe.preserve_overflow<>
#application = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>

// CHECK-LABEL: test_fn
func.func @test_fn() {
  return
}

// -----

#application = #lwe.application_data<message_type = i1>

// CHECK-LABEL: test_fn
func.func @test_fn() {
  return
}

// -----

#generator4 = #polynomial.int_polynomial<1 + x**1024>
#ring4 = #polynomial.ring<coefficientType = i32, coefficientModulus = 65536 : i32, polynomialModulus=#generator4>
#inverse_canonical_enc = #lwe.inverse_canonical_encoding<scaling_factor = 10000>

#plaintext_space = #lwe.plaintext_space<ring = #ring4, encoding = #inverse_canonical_enc>

// CHECK-LABEL: test_fn
func.func @test_fn() {
  return
}

// -----

#poly = #polynomial.int_polynomial<x**1024 + 1>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 12289 : i32, polynomialModulus=#poly>
#crt = #lwe.full_crt_packing_encoding<scaling_factor = 10000>
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #crt>

// CHECK-LABEL: test_fn
func.func @test_fn() {
  return
}

// -----

#key = #lwe.key<id = "1234">
#key_rlwe_rotate = #lwe.key<id = "1234", basis = 0, 2>
#key_rlwe_2 = #lwe.key<id = "1234", size = 2>

// CHECK-LABEL: test_fn
func.func @test_fn() {
  return
}

// -----

#generator4 = #polynomial.int_polynomial<1 + x**1024>
#ring4 = #polynomial.ring<coefficientType = i32, coefficientModulus = 65536 : i32, polynomialModulus=#generator4>

#ciphertext_space = #lwe.ciphertext_space<ring = #ring4, encryption_type = msb>

// CHECK-LABEL: test_fn
func.func @test_fn() {
  return
}
// -----

#modulus_chain = #lwe.modulus_chain<elements = <463187969 : i32, 33538049 : i32>, current = 0>

// CHECK-LABEL: test_fn
func.func @test_fn() {
  return
}
