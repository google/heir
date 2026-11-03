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
  // expected-error@below {{only lsb encryption is supported, got encryption type msb}}
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
#ciphertext_space_mix = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = mix>

!plaintext_rlwe = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ciphertext_mix = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_mix, key = #key>
!pk = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L0_1_x8_>

func.func @test_pk_encrypt_mix(%arg0: !plaintext_rlwe, %arg1: !pk) -> !ciphertext_mix {
  // expected-error@below {{only lsb encryption is supported, got encryption type mix}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_encrypt'}}
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!plaintext_rlwe, !pk) -> !ciphertext_mix
  return %0 : !ciphertext_mix
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
  // expected-error@below {{only lsb encryption is supported, got encryption type msb}}
  // expected-error@below {{failed to legalize operation 'lwe.rlwe_decrypt'}}
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ciphertext_msb, !sk) -> !plaintext_rlwe
  return %0 : !plaintext_rlwe
}
