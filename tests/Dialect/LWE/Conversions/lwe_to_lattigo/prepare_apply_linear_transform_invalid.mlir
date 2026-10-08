// RUN: heir-opt --split-input-file --lwe-to-lattigo --verify-diagnostics %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>
!prepared = !kernel.prepared_linear_transform<level = 0, slots = 512, log_bsgs_ratio = 0>

module attributes {backend.lattigo, scheme.ckks} {
  func.func @test_apply_equal_level(%ct: !ct, %lt: !prepared) -> !ct {
    // expected-error@below {{expected output level (0) to be exactly one less than input level (0)}}
    // expected-error@below {{failed to legalize}}
    %0 = kernel.apply_linear_transform %ct, %lt : !ct, !prepared -> !ct
    return %0 : !ct
  }
}
