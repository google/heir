// RUN: heir-opt --split-input-file --verify-diagnostics %s

#encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
#params = #lwe.lwe_params<cmod=7917, dimension=4>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

func.func @test_bad_coeff_len(%a: !ciphertext, %b: !ciphertext) -> () {
  // expected-error@+1 {{number of coefficients must match number of inputs}}
  %0 = cggi.lut_lincomb %a, %b {coefficients = array<i32: 1, 1, 1>, lookup_table = 68 : index} : !ciphertext
  return
}


// -----

#encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
#params = #lwe.lwe_params<cmod=7917, dimension=4>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

func.func @test_overflowing_coeff(%a: !ciphertext, %b: !ciphertext) -> () {
  // expected-error@below {{coefficient pushes error bits into message space}}
  // expected-note@below {{coefficient is 95}}
  // expected-note@below {{largest allowable coefficient is 7}}
  %0 = cggi.lut_lincomb %a, %b {coefficients = array<i32: 1, 95>, lookup_table = 68 : index} : !ciphertext
  return
}

// -----

#encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
#params = #lwe.lwe_params<cmod=7917, dimension=4>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

func.func @test_too_large_lut(%a: !ciphertext, %b: !ciphertext) -> () {
  // expected-error@below {{LUT is larger than available cleartext bit width}}
  // expected-note@below {{LUT has 18 active bits}}
  // expected-note@below {{max LUT size is 8 bits}}
  %0 = cggi.lut_lincomb %a, %b {coefficients = array<i32: 1, 1>, lookup_table = 172836 : index} : !ciphertext
  return
}
