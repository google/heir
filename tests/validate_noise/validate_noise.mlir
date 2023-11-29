// RUN: heir-opt --cggi-set-default-parameters --lwe-set-default-parameters --validate-noise %s

#encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
#poly = #polynomial.polynomial<1 + x**1024>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding>

// TODO(https://github.com/google/heir/issues/296): use lwe.encrypt with
// realistic initial noise.

// CHECK-LABEL: @test_defaults_are_valid_for_single_add
func.func @test_defaults_are_valid_for_single_add() -> !ciphertext {
  %0 = arith.constant 0 : i1
  %1 = arith.constant 1 : i1
  %2 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
  %3 = lwe.encode %1 { encoding = #encoding }: i1 to !plaintext
  %4 = lwe.trivial_encrypt %2 : !plaintext to !ciphertext
  %5 = lwe.trivial_encrypt %3 : !plaintext to !ciphertext
  %6 = lwe.add %4, %5 : !ciphertext
  return %6 : !ciphertext
}

// CHECK-LABEL: @test_boostrap_unknown_noise_input
func.func @test_boostrap_unknown_noise_input(%0 : !ciphertext) -> !ciphertext {
  %1 = cggi.lut2(%0, %0) {lookup_table = 1 : ui4} : !ciphertext
  return %1 : !ciphertext
}

// CHECK-LABEL: @test_add_post_bootstrap
func.func @test_add_post_bootstrap(%0 : !ciphertext) -> !ciphertext {
  %1 = cggi.lut2(%0, %0) {lookup_table = 1 : ui4} : !ciphertext
  %2 = lwe.add %1, %1 : !ciphertext
  return %2 : !ciphertext
}
