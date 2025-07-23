// RUN: heir-opt %s --lwe-to-polynomial --verify-diagnostics 2>&1

#key = #lwe.key<>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
!Z1095233372161 = !mod_arith.int<1095233372161 : i64>
!Z7917 = !mod_arith.int<7917:i32>
#ring = #polynomial.ring<coefficientType=!Z7917, polynomialModulus=#my_poly>
#ring_Z1095233372161 = #polynomial.ring<coefficientType = !Z1095233372161, polynomialModulus = #my_poly>
#full_crt_packing_encoding = #lwe.constant_coefficient_encoding<scaling_factor = 2>

#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #full_crt_packing_encoding>
#invalid_ciphertext_space = #lwe.ciphertext_space<ring = #ring_Z1095233372161, encryption_type = lsb, size=10>

!plaintext_rlwe = !lwe.new_lwe_plaintext<application_data = <message_type = f16>, plaintext_space = #plaintext_space>
!invalid_ciphertext = !lwe.new_lwe_ciphertext<application_data = <message_type = f16>, plaintext_space = #plaintext_space, ciphertext_space = #invalid_ciphertext_space, key = #key>

!rlwe_key = !lwe.new_lwe_secret_key<key = #key, ring = #ring_Z1095233372161>


func.func @invalid_dimensions_rlwe_decrypt(%arg0: !invalid_ciphertext, %arg1: !rlwe_key) -> !plaintext_rlwe {
  // expected-error@below {{Expected 2 dimensional ciphertext, found ciphertext tensor dimension = 10}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_decrypt'}}
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!invalid_ciphertext, !rlwe_key) -> !plaintext_rlwe
  return %0 : !plaintext_rlwe
}
