// RUN: heir-opt --split-input-file %s | FileCheck %s

// This simply tests for syntax.

#encoding = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>

// CHECK: test_valid_lwe_encode
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

// CHECK: test_valid_lwe_unspecified_encode
func.func @test_valid_lwe_unspecified_encode() {
    %0 = arith.constant 0 : i1
    // CHECK: unspecified_bit_field_encoding
    %2 = lwe.encode %0 { encoding = #unspecified_encoding }: i1 to !plaintext_unspecified
  return
}

// -----

#encoding0 = #lwe.unspecified_bit_field_encoding<
  cleartext_bitwidth=3>

// CHECK: test_valid_unspecified_lwe_attribute
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

// CHECK: test_valid_lwe_attribute
func.func @test_valid_lwe_attribute() -> tensor<2xi16, #encoding1> {
  %c0 = arith.constant 0 : index
  %two = arith.constant 2 : i16
  // CHECK: bit_field_encoding
  %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16, #encoding1>
  return %coeffs1 : tensor<2xi16, #encoding1>
}

// -----

#preserve_overflow = #lwe.preserve_overflow<>
#application = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>

// CHECK: test_fn
func.func @test_fn() {
  return
}

// -----

#application = #lwe.application_data<message_type = i1>

// CHECK: test_fn
func.func @test_fn() {
  return
}

// -----

#generator4 = #polynomial.int_polynomial<1 + x**1024>
#ring4 = #polynomial.ring<coefficientType=!mod_arith.int<65536:i32>, polynomialModulus=#generator4>
#inverse_canonical_enc = #lwe.inverse_canonical_encoding<scaling_factor = 10000>

#plaintext_space = #lwe.plaintext_space<ring = #ring4, encoding = #inverse_canonical_enc>

// CHECK: test_fn
func.func @test_fn() {
  return
}

// -----

#poly = #polynomial.int_polynomial<x**1024 + 1>
#ring = #polynomial.ring<coefficientType=!mod_arith.int<12289:i32>, polynomialModulus=#poly>
#crt = #lwe.full_crt_packing_encoding<scaling_factor = 10000>
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #crt>

// CHECK: test_fn
func.func @test_fn() {
  return
}

// -----

#key = #lwe.key<>
#key_rlwe_rotate = #lwe.key<slot_index = 2>
#key_rlwe_2 = #lwe.key<slot_index = 0>

// CHECK: test_fn
func.func @test_fn() {
  return
}

// -----

#generator4 = #polynomial.int_polynomial<1 + x**1024>
#ring4 = #polynomial.ring<coefficientType=!mod_arith.int<65536:i32>, polynomialModulus=#generator4>

#ciphertext_space = #lwe.ciphertext_space<ring = #ring4, encryption_type = msb>

// CHECK: test_fn
func.func @test_fn() {
  return
}
// -----

#modulus_chain = #lwe.modulus_chain<elements = <463187969 : i32, 33538049 : i32>, current = 0>

// CHECK: test_fn
func.func @test_fn() {
  return
}
