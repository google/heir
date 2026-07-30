// RUN: heir-opt --verify-diagnostics %s

!Z1073741441_i64 = !mod_arith.int<1073741441 : i64>
!Z536870273_i64 = !mod_arith.int<536870273 : i64>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 29>
#key = #lwe.key<>
#modulus_chain_L0 = #lwe.modulus_chain<elements = <1073741441 : i64, 536870273 : i64>, current = 0>
#modulus_chain_L1 = #lwe.modulus_chain<elements = <1073741441 : i64, 536870273 : i64>, current = 1>
#ring_f64_1_x8 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
!rns_L0 = !rns.rns<!Z1073741441_i64>
!rns_L1 = !rns.rns<!Z1073741441_i64, !Z536870273_i64>
#ring_rns_L0_1_x8 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**8>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x8, encryption_type = mix>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #encoding>>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1>

module {
  func.func @add_plain_mismatched_rings(
      %ctx: !jaxiteword.crypto_context<>, %ct: !ct_L1, %pt: !pt) -> !ct_L0 {
    // expected-error@+1 {{requires all operands and results to have the same rings}}
    %0 = jaxiteword.add_plain %ctx, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L0
    return %0 : !ct_L0
  }
}
