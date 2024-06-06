// RUN: not heir-opt --verify-diagnostics --split-input-file %s 2>&1 | FileCheck %s

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

#different_encoding = #lwe.bit_field_encoding<
  cleartext_start=16,
  cleartext_bitwidth=3>
!mismatch_plaintext = !lwe.lwe_plaintext<encoding = #different_encoding>

// expected-error@below {{encoding attr must match output LWE plaintext encoding}}
func.func @test_invalid_lwe_encode() {
    %0 = arith.constant 0 : i1
    %2 = lwe.encode %0 { encoding = #encoding }: i1 to !mismatch_plaintext
  return
}

// -----
#params = #lwe.lwe_params<coefficientType = i32, coefficientModulus = 7917 : i32, dimension=10>
#mismatch_params = #lwe.lwe_params<coefficientType = i32, coefficientModulus = 7917 : i32, dimension=11>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

// expected-error@below {{LWE params attr must match output LWE ciphertext LWE params attr}}
func.func @test_invalid_lwe_trivial_encrypt_params() {
    %0 = arith.constant 0 : i1
    %1 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
    %2 = lwe.trivial_encrypt %1 { lwe_params = #mismatch_params }: !plaintext to !ciphertext
  return
}

// -----

!mismatch_ciphertext = !lwe.lwe_ciphertext<encoding = #different_encoding, lwe_params = #params>

// expected-error@below {{LWE plaintext encoding must match output LWE ciphertext encoding attr}}
func.func @test_invalid_lwe_trivial_encrypt_encoding() {
    %0 = arith.constant 0 : i1
    %1 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
    %2 = lwe.trivial_encrypt %1 { lwe_params = #params }: !plaintext to !mismatch_ciphertext
  return
}
