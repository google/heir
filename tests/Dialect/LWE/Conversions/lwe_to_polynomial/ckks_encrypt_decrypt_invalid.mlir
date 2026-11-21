// RUN: heir-opt %s --lwe-to-polynomial --verify-diagnostics --split-input-file

// CKKS encryption and decryption require the plaintext to already be encoded
// into the ring of its ciphertext.

// A plaintext with non-modular coefficients.

!Zq = !mod_arith.int<1095233372161 : i64>
#key = #lwe.key<>
#ring_Q = #polynomial.ring<coefficientType = !rns.rns<!Zq>, polynomialModulus = <1 + x**8>>
#ring_i64 = #polynomial.ring<coefficientType = i64, polynomialModulus = <1 + x**8>>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_i64, encoding = #encoding>>
!ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_Q, encryption_type = ckks>, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_Q>

func.func @test_sk_encrypt_non_modular_plaintext(%arg0: !pt, %arg1: !sk) -> !ct {
  // expected-error@below {{expected plaintext to have modular (e.g., RNS) polynomial coefficients, got 'i64'}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_encrypt'}}
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!pt, !sk) -> !ct
  return %0 : !ct
}

// -----

// A plaintext with a different RNS basis than its ciphertext.

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
#key = #lwe.key<>
#ring_Q = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**8>>
#ring_Q1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1>, polynomialModulus = <1 + x**8>>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_Q1, encoding = #encoding>>
!ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_Q, encryption_type = ckks>, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_Q>

func.func @test_decrypt_mismatched_basis(%arg0: !ct, %arg1: !sk) -> !pt {
  // expected-error@below {{expected plaintext and ciphertext to have the same ring}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_decrypt'}}
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ct, !sk) -> !pt
  return %0 : !pt
}

// -----

// A plaintext with the same RNS basis as its ciphertext, but a different
// polynomial modulus.

!Zq = !mod_arith.int<1095233372161 : i64>
#key = #lwe.key<>
#ring_Q = #polynomial.ring<coefficientType = !rns.rns<!Zq>, polynomialModulus = <1 + x**8>>
#ring_Q_x16 = #polynomial.ring<coefficientType = !rns.rns<!Zq>, polynomialModulus = <1 + x**16>>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_Q_x16, encoding = #encoding>>
!ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_Q, encryption_type = ckks>, key = #key>
!pk = !lwe.lwe_public_key<key = #key, ring = #ring_Q>

func.func @test_pk_encrypt_mismatched_polynomial_modulus(%arg0: !pt, %arg1: !pk) -> !ct {
  // expected-error@below {{expected plaintext and ciphertext to have the same ring}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_encrypt'}}
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!pt, !pk) -> !ct
  return %0 : !ct
}
