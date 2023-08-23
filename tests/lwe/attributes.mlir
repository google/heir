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
