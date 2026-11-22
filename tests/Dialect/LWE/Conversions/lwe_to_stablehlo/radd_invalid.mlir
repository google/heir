// RUN: heir-opt --lwe-to-stablehlo --verify-diagnostics --split-input-file %s

!Z1095233372161 = !mod_arith.int<1095233372161 : i64>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#ring_wide = #polynomial.ring<coefficientType = !Z1095233372161, polynomialModulus = <1 + x**16>>
#ct_space_wide = #lwe.ciphertext_space<ring = #ring_wide, encryption_type = ckks>
!ct_wide = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = #ct_space_wide, key = #key>

module {
  func.func @modulus_too_wide(%ct0: !ct_wide, %ct1: !ct_wide) -> !ct_wide {
    // expected-error @+1 {{ui32 StableHLO lowering requires moduli in (1, 2^32)}}
    %sum = lwe.radd %ct0, %ct1 : (!ct_wide, !ct_wide) -> !ct_wide
    return %sum : !ct_wide
  }
}

// -----

!Z65537 = !mod_arith.int<65537 : i64>
#encoding_nonpow2 = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key_nonpow2 = #lwe.key<>
#ring_6 = #polynomial.ring<coefficientType = !Z65537, polynomialModulus = <1 + x**6>>
#ct_space_6 = #lwe.ciphertext_space<ring = #ring_6, encryption_type = ckks>
!ct_6 = !lwe.lwe_ciphertext<encoding = #encoding_nonpow2, ciphertext_space = #ct_space_6, key = #key_nonpow2>

module {
  func.func @degree_not_power_of_two(%ct0: !ct_6, %ct1: !ct_6) -> !ct_6 {
    // expected-error @+1 {{ciphertext polynomial degree must be a power of two, got 6}}
    %sum = lwe.radd %ct0, %ct1 : (!ct_6, !ct_6) -> !ct_6
    return %sum : !ct_6
  }
}

// -----

!Z65537_no_mod = !mod_arith.int<65537 : i64>
#encoding_no_mod = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key_no_mod = #lwe.key<>
#ring_no_mod = #polynomial.ring<coefficientType = !Z65537_no_mod>
#ct_space_no_mod = #lwe.ciphertext_space<ring = #ring_no_mod, encryption_type = ckks>
!ct_no_mod = !lwe.lwe_ciphertext<encoding = #encoding_no_mod, ciphertext_space = #ct_space_no_mod, key = #key_no_mod>

module {
  func.func @missing_polynomial_modulus(%ct0: !ct_no_mod, %ct1: !ct_no_mod) -> !ct_no_mod {
    // expected-error @+1 {{ciphertext ring must specify a polynomial modulus}}
    %sum = lwe.radd %ct0, %ct1 : (!ct_no_mod, !ct_no_mod) -> !ct_no_mod
    return %sum : !ct_no_mod
  }
}

// -----

!Z2147473409 = !mod_arith.int<2147473409 : i64>
#encoding_unranked = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key_unranked = #lwe.key<>
#ring_16 = #polynomial.ring<coefficientType = !Z2147473409, polynomialModulus = <1 + x**16>>
#ct_space_16 = #lwe.ciphertext_space<ring = #ring_16, encryption_type = ckks>
!ct_16 = !lwe.lwe_ciphertext<encoding = #encoding_unranked, ciphertext_space = #ct_space_16, key = #key_unranked>

module {
  func.func @unranked_ciphertext_tensor(%t: tensor<*x!ct_16>) -> tensor<*x!ct_16> {
    // expected-error @+1 {{expected 1D statically shaped tensor of LWE ciphertexts}}
    return %t : tensor<*x!ct_16>
  }
}
