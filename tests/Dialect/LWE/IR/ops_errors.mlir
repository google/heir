// RUN: heir-opt --verify-diagnostics --split-input-file %s 2>&1

#encoding = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>

#different_encoding = #lwe.bit_field_encoding<
  cleartext_start=16,
  cleartext_bitwidth=3>
!mismatch_plaintext = !lwe.lwe_plaintext<encoding = #different_encoding>

func.func @test_invalid_lwe_encode() {
    %0 = arith.constant 0 : i1
    // expected-error@below {{failed to verify that the first arg's type's encoding matches the given encoding}}
    %2 = lwe.encode %0 { encoding = #encoding }: i1 to !mismatch_plaintext
  return
}

// -----

#encoding = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>

#params = #lwe.lwe_params<cmod = 7917 : i32, dimension=10>
#mismatch_params = #lwe.lwe_params<cmod = 7917 : i32, dimension=11>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

func.func @test_invalid_lwe_trivial_encrypt_params() {
    %0 = arith.constant 0 : i1
    %1 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
    // expected-error@below {{lwe_params attr must match on the op and the output type}}
    %2 = lwe.trivial_encrypt %1 { lwe_params = #mismatch_params }: !plaintext to !ciphertext
  return
}

// -----

#encoding = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>
#different_encoding = #lwe.bit_field_encoding<
  cleartext_start=16,
  cleartext_bitwidth=3>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>
#params = #lwe.lwe_params<cmod = 7917 : i32, dimension=10>
!mismatch_ciphertext = !lwe.lwe_ciphertext<encoding = #different_encoding, lwe_params = #params>

func.func @test_invalid_lwe_trivial_encrypt_encoding() {
    %0 = arith.constant 0 : i1
    %1 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
    // expected-error@below {{op failed to verify that the first arg's type's encoding matches the given encoding}}
    %2 = lwe.trivial_encrypt %1 { lwe_params = #params }: !plaintext to !mismatch_ciphertext
  return
}
