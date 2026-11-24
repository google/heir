// RUN: heir-opt %s --lwe-to-polynomial --verify-diagnostics --split-input-file

// BGV (lsb) encryption and decryption require the plaintext coefficient type
// to be a mod_arith type, whose modulus is the plaintext modulus.

!Zq = !mod_arith.int<1095233372161 : i64>
#key = #lwe.key<>
#ring_Q = #polynomial.ring<coefficientType = !rns.rns<!Zq>, polynomialModulus = <1 + x**8>>
#ring_i64 = #polynomial.ring<coefficientType = i64, polynomialModulus = <1 + x**8>>
#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_i64, encoding = #encoding>>
!ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_Q, encryption_type = lsb>, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_Q>

func.func @test_sk_encrypt_non_mod_arith_plaintext(%arg0: !pt, %arg1: !sk) -> !ct {
  // expected-error@below {{expected plaintext to have mod_arith polynomial coefficients}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_encrypt'}}
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!pt, !sk) -> !ct
  return %0 : !ct
}

// -----

!Zq = !mod_arith.int<1095233372161 : i64>
#key = #lwe.key<>
#ring_Q = #polynomial.ring<coefficientType = !rns.rns<!Zq>, polynomialModulus = <1 + x**8>>
#ring_i64 = #polynomial.ring<coefficientType = i64, polynomialModulus = <1 + x**8>>
#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_i64, encoding = #encoding>>
!ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_Q, encryption_type = lsb>, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_Q>

func.func @test_decrypt_non_mod_arith_plaintext(%arg0: !ct, %arg1: !sk) -> !pt {
  // expected-error@below {{expected plaintext to have mod_arith polynomial coefficients}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_decrypt'}}
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ct, !sk) -> !pt
  return %0 : !pt
}
