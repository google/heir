// RUN: heir-opt %s --lwe-to-polynomial --verify-diagnostics 2>&1

#encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#invalid_rlwe_params = #lwe.rlwe_params<dimension=10, ring=#ring>
!plaintext_rlwe = !lwe.rlwe_plaintext<encoding = #encoding, ring = #ring, underlying_type=i3>
!invalid_ciphertext = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #invalid_rlwe_params, underlying_type=i3>
!invalid_secret_key = !lwe.rlwe_secret_key<rlwe_params=#invalid_rlwe_params>

func.func @invalid_dimensions_rlwe_decrypt(%arg0: !invalid_ciphertext, %arg1: !invalid_secret_key) -> !plaintext_rlwe {
  // expected-error@below {{Expected 2 dimensional ciphertext, found ciphertext tensor dimension = 10}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_decrypt'}}
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!invalid_ciphertext, !invalid_secret_key) -> !plaintext_rlwe
  return %0 : !plaintext_rlwe
}
