// RUN: heir-opt --split-input-file %s | FileCheck %s

// This simply tests for syntax.

#poly = #polynomial.int_polynomial<x>
#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.new_lwe_plaintext<application_data = <message_type = i1, overflow = #preserve_overflow>, plaintext_space = #pspace>

// CHECK: test_valid_lwe_encode
func.func @test_valid_lwe_encode() {
    %0 = arith.constant 0 : i1
    // CHECK: constant_coefficient_encoding
    %2 = lwe.encode %0 { overflow = #preserve_overflow, plaintext_bits = 3 : index }: i1 to !plaintext
  return
}

// -----

#poly = #polynomial.int_polynomial<x>
#no_overflow = #lwe.no_overflow<>
#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext_nooverflow = !lwe.new_lwe_plaintext<application_data = <message_type = i1>, plaintext_space = #pspace>

// CHECK: test_valid_lwe_default_overflow
func.func @test_valid_lwe_default_overflow() {
    %0 = arith.constant 0 : i1
    // CHECK: no_overflow
    %2 = lwe.encode %0 { overflow = #no_overflow, plaintext_bits = 3 : index }: i1 to !plaintext_nooverflow
  return
}

// -----

#poly = #polynomial.int_polynomial<x>
#key = #lwe.key<slot_index = 0>
!modarith = !mod_arith.int<12289:i32>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = !modarith, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext_modarith = !lwe.new_lwe_plaintext<application_data = <message_type = i4>, plaintext_space = #pspace>

// CHECK: test_valid_lwe_modarith_type
func.func @test_valid_lwe_modarith_type(%arg0: !plaintext_modarith) {
  return
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
