// RUN: heir-opt %s --lwe-to-polynomial --verify-diagnostics --split-input-file

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#key = #lwe.key<>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_Z65537_i64_1_x8_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**8>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 2>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x8_, encoding = #full_crt_packing_encoding>
#ciphertext_space_msb = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = msb>

!plaintext_rlwe = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ciphertext_msb = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_msb, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>

func.func @test_sk_encrypt_msb(%arg0: !plaintext_rlwe, %arg1: !sk) -> !ciphertext_msb {
  // expected-error@below {{only lsb and ckks encryption are supported, got encryption type msb}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_encrypt'}}
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!plaintext_rlwe, !sk) -> !ciphertext_msb
  return %0 : !ciphertext_msb
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#key = #lwe.key<>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_Z65537_i64_1_x8_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**8>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 2>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x8_, encoding = #full_crt_packing_encoding>
#ciphertext_space_msb = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = msb>

!plaintext_rlwe = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ciphertext_msb = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_msb, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>

func.func @test_decrypt_msb(%arg0: !ciphertext_msb, %arg1: !sk) -> !plaintext_rlwe {
  // expected-error@below {{only lsb and ckks encryption are supported, got encryption type msb}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_decrypt'}}
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ciphertext_msb, !sk) -> !plaintext_rlwe
  return %0 : !plaintext_rlwe
}

// -----

// CKKS plaintexts must have modular (e.g., RNS) polynomial coefficients.

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
#key = #lwe.key<>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_i64_1_x8_ = #polynomial.ring<coefficientType = i64, polynomialModulus = <1 + x**8>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#plaintext_space = #lwe.plaintext_space<ring = #ring_i64_1_x8_, encoding = #inverse_canonical_encoding>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = ckks>

!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>

func.func @test_ckks_sk_encrypt_non_modular_plaintext(%arg0: !pt, %arg1: !sk) -> !ct {
  // expected-error@below {{expected plaintext to have modular (e.g., RNS) polynomial coefficients, got}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_encrypt'}}
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!pt, !sk) -> !ct
  return %0 : !ct
}

// -----

// CKKS plaintexts must have the same RNS basis as the ciphertext they are
// decrypted from.

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
#key = #lwe.key<>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**8>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#plaintext_space_L1_ = #lwe.plaintext_space<ring = #ring_rns_L1_1_x8_, encoding = #inverse_canonical_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = ckks>

!pt_L1 = !lwe.lwe_plaintext<plaintext_space = #plaintext_space_L1_>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space_L1_, ciphertext_space = #ciphertext_space_L0_, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>

func.func @test_ckks_decrypt_mismatched_basis(%arg0: !ct, %arg1: !sk) -> !pt_L1 {
  // expected-error@below {{expected plaintext and ciphertext to have the same RNS basis}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_decrypt'}}
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ct, !sk) -> !pt_L1
  return %0 : !pt_L1
}

// -----

// CKKS plaintexts must have the same RNS basis as the ciphertext they are
// encrypted to.

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
#key = #lwe.key<>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**8>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#plaintext_space_L1_ = #lwe.plaintext_space<ring = #ring_rns_L1_1_x8_, encoding = #inverse_canonical_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = ckks>

!pt_L1 = !lwe.lwe_plaintext<plaintext_space = #plaintext_space_L1_>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space_L1_, ciphertext_space = #ciphertext_space_L0_, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>

func.func @test_ckks_sk_encrypt_mismatched_basis(%arg0: !pt_L1, %arg1: !sk) -> !ct {
  // expected-error@below {{expected plaintext and ciphertext to have the same RNS basis}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_encrypt'}}
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!pt_L1, !sk) -> !ct
  return %0 : !ct
}

// -----

// CKKS plaintexts must have the same RNS basis as the ciphertext they are
// encrypted to.

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
#key = #lwe.key<>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**8>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#plaintext_space_L1_ = #lwe.plaintext_space<ring = #ring_rns_L1_1_x8_, encoding = #inverse_canonical_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = ckks>

!pt_L1 = !lwe.lwe_plaintext<plaintext_space = #plaintext_space_L1_>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space_L1_, ciphertext_space = #ciphertext_space_L0_, key = #key>
!pk = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L0_1_x8_>

func.func @test_ckks_pk_encrypt_mismatched_basis(%arg0: !pt_L1, %arg1: !pk) -> !ct {
  // expected-error@below {{expected plaintext and ciphertext to have the same RNS basis}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_encrypt'}}
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!pt_L1, !pk) -> !ct
  return %0 : !ct
}
