// RUN: heir-opt %s 2>&1 | FileCheck %s

// This simply tests for syntax.

#encoding = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>
#params = #lwe.lwe_params<cmod=7917, dimension=10>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>


// CHECK-LABEL: test_valid_lwe_ciphertext
func.func @test_valid_lwe_ciphertext(%arg0 : !ciphertext) -> !ciphertext {
  return %arg0 : !ciphertext
}


!ciphertext_noparams = !lwe.lwe_ciphertext<encoding = #encoding>

// CHECK-LABEL: test_valid_lwe_ciphertext_unspecified
func.func @test_valid_lwe_ciphertext_unspecified(%arg0 : !ciphertext_noparams) -> !ciphertext_noparams {
  return %arg0 : !ciphertext_noparams
}


#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#rlwe_params = #lwe.rlwe_params<dimension=10, ring=#ring>
!ciphertext_rlwe = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #rlwe_params, underlying_type=i3>

// CHECK-LABEL: test_valid_rlwe_ciphertext
func.func @test_valid_rlwe_ciphertext(%arg0 : !ciphertext_rlwe) -> !ciphertext_rlwe {
  return %arg0 : !ciphertext_rlwe
}

#key = #lwe.key<id = "1234", size = 1>
!secret_key = !lwe.new_lwe_secret_key<key = #key, ring = #ring>

// CHECK-LABEL: test_new_lwe_secret_key
func.func @test_new_lwe_secret_key(%arg0 : !secret_key) -> !secret_key {
  return %arg0 :!secret_key
}

!public_key = !lwe.new_lwe_public_key<key = #key, ring = #ring>

// CHECK-LABEL: test_new_lwe_public_key
func.func @test_new_lwe_public_key(%arg0 : !public_key) -> !public_key {
  return %arg0 : !public_key
}

#key_mult = #lwe.key<id = "1234", size = 1, basis = 2>
#key_rotate = #lwe.key<id = "1234", size = 1, rotate = 2>

#keyswitch_bv_base = #lwe.bv_keyswitch_technique<base = 65536, dnum = 0>
#keyswitch_bv = #lwe.bv_keyswitch_technique<base = 0, dnum = 3>
#keyswitch_ghs = #lwe.ghs_keyswitch_technique<extra_modulus=<elements = <65537 : i32>, current = 0>>

!evaluation_key_mult = !lwe.new_lwe_evaluation_key<from_key=#key_mult, to_key=#key, ring=#ring, keyswitch_techniques= #keyswitch_bv, #keyswitch_ghs>
!evaluation_key_rotate = !lwe.new_lwe_evaluation_key<from_key=#key_rotate, to_key=#key, ring=#ring, keyswitch_techniques= #keyswitch_bv_base>

// CHECK-LABEL test_new_lwe_evaluation_key_mult

func.func @test_new_lwe_evaluation_key_mult(%arg0 : !evaluation_key_mult) -> !evaluation_key_mult {
  return %arg0 : !evaluation_key_mult
}

// CHECK-LABEL test_new_lwe_evaluation_key_rotate

func.func @test_new_lwe_evaluation_key_rotate(%arg0 : !evaluation_key_rotate) -> !evaluation_key_rotate {
  return %arg0 : !evaluation_key_rotate
}

#preserve_overflow = #lwe.preserve_overflow<>
#application_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#inverse_canonical_enc = #lwe.inverse_canonical_encoding<scaling_factor = 10000>
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #inverse_canonical_enc>

!new_lwe_plaintext = !lwe.new_lwe_plaintext<application_data = #application_data, plaintext_space = #plaintext_space>

// CHECK-LABEL: test_new_lwe_plaintext
func.func @test_new_lwe_plaintext(%arg0 : !new_lwe_plaintext) -> !new_lwe_plaintext {
  return %arg0 : !new_lwe_plaintext
}

#ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = msb>
#modulus_chain = #lwe.modulus_chain<elements = <7917 : i32>, current = 0>

!new_lwe_ciphertext = !lwe.new_lwe_ciphertext<application_data = #application_data, plaintext_space = #plaintext_space, key = #key, ciphertext_space = #ciphertext_space, modulus_chain = #modulus_chain>

// CHECK-LABEL: test_new_lwe_ciphertext
func.func @test_new_lwe_ciphertext(%arg0 : !new_lwe_ciphertext) -> !new_lwe_ciphertext {
  return %arg0 : !new_lwe_ciphertext
}
