// RUN: heir-opt --split-input-file --verify-diagnostics %s

#poly = #polynomial.int_polynomial<x>
#key = #lwe.key<slot_index = 0>
#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>,
  encryption_type = msb, size = 742>
!ciphertext = !lwe.lwe_ciphertext<encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456, plaintext_bits = 4>, ciphertext_space = #cspace, key = #key>

func.func @test_bad_coeff_len(%a: !ciphertext, %b: !ciphertext) -> () {
  // expected-error@+1 {{number of coefficients must match number of inputs}}
  %0 = cggi.lut_lincomb %a, %b <coefficients = [1, 1, 1], lookup_table = 68 : index> : !ciphertext
  return
}


// -----

#poly = #polynomial.int_polynomial<x>
#key = #lwe.key<slot_index = 0>
#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>,
  encryption_type = msb, size = 742>
!ciphertext = !lwe.lwe_ciphertext<encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456, plaintext_bits = 3>, ciphertext_space = #cspace, key = #key>

func.func @test_overflowing_coeff(%a: !ciphertext, %b: !ciphertext) -> () {
  // expected-error@below {{coefficient pushes error bits into message space}}
  // expected-note@below {{coefficient is 95}}
  // expected-note@below {{largest allowable coefficient is 7}}
  %0 = cggi.lut_lincomb %a, %b <coefficients = [1, 95], lookup_table = 68 : index> : !ciphertext
  return
}

// -----

#poly = #polynomial.int_polynomial<x>
#key = #lwe.key<slot_index = 0>
#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>,
  encryption_type = msb, size = 742>
!ciphertext = !lwe.lwe_ciphertext<encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456, plaintext_bits = 3>, ciphertext_space = #cspace, key = #key>

func.func @test_too_large_lut(%a: !ciphertext, %b: !ciphertext) -> () {
  // expected-error@below {{LUT is larger than available cleartext bit width}}
  // expected-note@below {{LUT has 18 active bits}}
  // expected-note@below {{max LUT size is 8 bits}}
  %0 = cggi.lut_lincomb %a, %b <coefficients = [1, 1], lookup_table = 172836 : index> : !ciphertext
  return
}
