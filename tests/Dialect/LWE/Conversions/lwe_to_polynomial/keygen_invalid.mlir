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

// -----

// BGV key-switching keys need noise that is a multiple of the plaintext
// modulus, which gen_relin_key cannot express yet.
!Zq0 = !mod_arith.int<17 : i32>
!Zp0 = !mod_arith.int<97 : i32>
!rns_Q = !rns.rns<!Zq0>
!rns_QP = !rns.rns<!Zq0, !Zp0>
#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**8>>
#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**8>>
#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring_Q>
!ksk_ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_QP, encryption_type = lsb>, key = #lwe.key<>>

func.func @test_gen_relin_key_bgv(%sk: !sk) -> tensor<1x!ksk_ct> {
  // expected-error@below {{lsb (BGV) relinearization keys are not supported}}
  // expected-error@below {{failed to legalize operation 'lwe.gen_relin_key'}}
  %rlk = lwe.gen_relin_key %sk : (!sk) -> tensor<1x!ksk_ct>
  return %rlk : tensor<1x!ksk_ct>
}
