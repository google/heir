// RUN: heir-opt --split-input-file --cggi-set-default-parameters --lwe-set-default-parameters --validate-noise --verify-diagnostics %s

// TODO(https://github.com/google/heir/issues/296): use lwe.encrypt with
// realistic initial noise.

// #encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
// #poly = #polynomial.polynomial<1 + x**1024>
// !plaintext = !lwe.lwe_plaintext<encoding = #encoding>
// !ciphertext = !lwe.lwe_ciphertext<encoding = #encoding>
//
// func.func @test_cant_add_unknown_value(%arg0 : !ciphertext) -> !ciphertext {
//   // expected-error@below {{uses SSA value with unknown noise variance}}
//   %1 = lwe.add %arg0, %arg0 : !ciphertext
//   return %1 : !ciphertext
// }
//
// // -----

#encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
#poly = #polynomial.polynomial<1 + x**1024>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding>

func.func @unknown_value_from_loop_result() -> !ciphertext {
  %0 = arith.constant 0 : i1
  %2 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
  %3 = lwe.trivial_encrypt %2 : !plaintext to !ciphertext

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  %5 = scf.for %arg1 = %c1 to %c5 step %c1 iter_args(%iter_arg = %3) -> !ciphertext {
    // expected-error@below {{uses SSA value with unknown noise variance}}
    %6 = lwe.add %iter_arg, %iter_arg : !ciphertext
    scf.yield %6 : !ciphertext
  }
  return %5 : !ciphertext
}
