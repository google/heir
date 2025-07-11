// RUN: heir-opt --verify-diagnostics --split-input-file %s 2>&1

#poly = #polynomial.int_polynomial<x>
#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.new_lwe_plaintext<application_data = <message_type = i1>, plaintext_space = #pspace>

// CHECK: test_invalid_overflow
func.func @test_invalid_overflow() {
    %0 = arith.constant 0 : i1
    // expected-error@below {{output type overflow must match overflow parameter}}
    %2 = lwe.encode %0 { overflow = #preserve_overflow, plaintext_bits = 3 : index }: i1 to !plaintext
  return
}

// -----

#poly = #polynomial.int_polynomial<x>
#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.new_lwe_plaintext<application_data = <message_type = i2, overflow = #preserve_overflow>, plaintext_space = #pspace>

// CHECK: test_invalid_application_data
func.func @test_invalid_application_data() {
    %0 = arith.constant 0 : i1
    // expected-error@below {{output type application data must match input type}}
    %2 = lwe.encode %0 { overflow = #preserve_overflow, plaintext_bits = 3 : index }: i1 to !plaintext
  return
}

// -----

#poly = #polynomial.int_polynomial<x>
#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.new_lwe_plaintext<application_data = <message_type = i1, overflow = #preserve_overflow>, plaintext_space = #pspace>

// CHECK: test_invalid_plaintext_bits
func.func @test_invalid_plaintext_bits() {
    %0 = arith.constant 0 : i1
    // expected-error@below {{LWE plaintext ring coefficient type width must match message bits parameter, expected 4 but got 3}}
    %2 = lwe.encode %0 { overflow = #preserve_overflow, plaintext_bits = 4 : index }: i1 to !plaintext
  return
}


// -----

#poly = #polynomial.int_polynomial<x>
#preserve_overflow = #lwe.preserve_overflow<>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.new_lwe_plaintext<application_data = <message_type = i1, overflow = #preserve_overflow>, plaintext_space = #pspace>

#key = #lwe.key<slot_index = 0>
#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>,
  encryption_type = msb, size = 742>
!ciphertext = !lwe.new_lwe_ciphertext<application_data = <message_type = i1, overflow = #preserve_overflow>, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

func.func @test_invalid_ciphertext_bits(%1: !plaintext) {
    // expected-error@below {{ciphertext modulus of the output must match the ciphertext_bits parameter, expected 64 but found 32}}
    %2 = lwe.trivial_encrypt %1 { ciphertext_bits = 64 : index }: !plaintext to !ciphertext
  return
}

// -----

#poly = #polynomial.int_polynomial<x>
#preserve_overflow = #lwe.preserve_overflow<>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.new_lwe_plaintext<application_data = <message_type = i1, overflow = #preserve_overflow>, plaintext_space = #pspace>

#key = #lwe.key<slot_index = 0>
#diff_pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 512>>

#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>,
  encryption_type = msb, size = 742>
!ciphertext = !lwe.new_lwe_ciphertext<application_data = <message_type = i1, overflow = #preserve_overflow>, plaintext_space = #diff_pspace, ciphertext_space = #cspace, key = #key>

func.func @test_invalid_mismatch_encoding(%1: !plaintext) {
    // expected-error@below {{op failed to verify that the first arg's type's encoding matches the given encoding}}
    %2 = lwe.trivial_encrypt %1 { ciphertext_bits = 32 : index }: !plaintext to !ciphertext
  return
}
