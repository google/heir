// RUN: heir-opt --verify-diagnostics --split-input-file %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!rns_Q0 = !rns.rns<!Zq0>
!rns_Q = !rns.rns<!Zq0, !Zq1>
#ring_Q0 = #polynomial.ring<coefficientType = !rns_Q0, polynomialModulus = <1 + x**1024>>
#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring_Q>
!pk_Q0 = !lwe.lwe_public_key<key = #lwe.key<>, ring = #ring_Q0>

func.func @test_pk_ring_mismatch(%sk: !sk) -> !pk_Q0 {
  // expected-error@+1 {{failed to verify that the keys' rings match}}
  %pk = lwe.gen_public_key %sk : (!sk) -> !pk_Q0
  return %pk : !pk_Q0
}

// -----

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zp0 = !mod_arith.int<998595133441 : i64>
!rns_Q = !rns.rns<!Zq0, !Zq1>
!rns_QP = !rns.rns<!Zq0, !Zq1, !Zp0>
#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**1024>>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring_Q>
!ksk_ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_QP, encryption_type = ckks>, key = #lwe.key<>>

func.func @test_wrong_num_digits(%sk: !sk) -> tensor<1x!ksk_ct> {
  // expected-error@+1 {{expected 2 key-switching key ciphertexts}}
  %rlk = lwe.gen_relin_key %sk : (!sk) -> tensor<1x!ksk_ct>
  return %rlk : tensor<1x!ksk_ct>
}

// -----

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!rns_Q = !rns.rns<!Zq0, !Zq1>
#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring_Q>
!ksk_ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_Q, encryption_type = ckks>, key = #lwe.key<>>

func.func @test_no_key_switching_primes(%sk: !sk) -> tensor<1x!ksk_ct> {
  // expected-error@+1 {{followed by at least one key-switching prime}}
  %rlk = lwe.gen_relin_key %sk : (!sk) -> tensor<1x!ksk_ct>
  return %rlk : tensor<1x!ksk_ct>
}

// -----

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zp0 = !mod_arith.int<998595133441 : i64>
!rns_Q = !rns.rns<!Zq0, !Zq1>
!rns_PQ = !rns.rns<!Zp0, !Zq0, !Zq1>
#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
#ring_PQ = #polynomial.ring<coefficientType = !rns_PQ, polynomialModulus = <1 + x**1024>>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring_Q>
!ksk_ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_PQ, encryption_type = ckks>, key = #lwe.key<>>

func.func @test_q_not_prefix(%sk: !sk) -> tensor<2x!ksk_ct> {
  // expected-error@+1 {{secret key's RNS basis followed by}}
  %rlk = lwe.gen_relin_key %sk : (!sk) -> tensor<2x!ksk_ct>
  return %rlk : tensor<2x!ksk_ct>
}

// -----

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zp0 = !mod_arith.int<998595133441 : i64>
!rns_Q = !rns.rns<!Zq0>
!rns_QP = !rns.rns<!Zq0, !Zp0>
#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**2048>>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring_Q>
!ksk_ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_QP, encryption_type = ckks>, key = #lwe.key<>>

func.func @test_poly_modulus_mismatch(%sk: !sk) -> tensor<1x!ksk_ct> {
  // expected-error@+1 {{same polynomial modulus}}
  %rlk = lwe.gen_relin_key %sk : (!sk) -> tensor<1x!ksk_ct>
  return %rlk : tensor<1x!ksk_ct>
}

// -----

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zp0 = !mod_arith.int<998595133441 : i64>
!rns_Q = !rns.rns<!Zq0>
!rns_QP = !rns.rns<!Zq0, !Zp0>
#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**1024>>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring_Q>
!ksk_ct_D3 = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_QP, encryption_type = ckks, size = 3>, key = #lwe.key<>>

func.func @test_ciphertext_size(%sk: !sk) -> tensor<1x!ksk_ct_D3> {
  // expected-error@+1 {{expected relinearization key ciphertexts of size 2}}
  %rlk = lwe.gen_relin_key %sk : (!sk) -> tensor<1x!ksk_ct_D3>
  return %rlk : tensor<1x!ksk_ct_D3>
}
