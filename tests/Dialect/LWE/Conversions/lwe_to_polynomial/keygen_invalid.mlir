// RUN: heir-opt --lwe-to-polynomial --verify-diagnostics --split-input-file %s

// The mask of a public key is sampled uniformly from R_Q, which requires
// modular coefficients.
#ring = #polynomial.ring<coefficientType = i32, polynomialModulus = <1 + x**8>>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring>
!pk = !lwe.lwe_public_key<key = #lwe.key<>, ring = #ring>

func.func @test_gen_public_key_non_modular_ring(%sk: !sk) -> !pk {
  // expected-error@below {{expected a ring with modular (e.g., RNS) coefficients and a polynomial modulus}}
  // expected-error@below {{failed to legalize operation 'lwe.gen_public_key'}}
  %pk = lwe.gen_public_key %sk : (!sk) -> !pk
  return %pk : !pk
}
