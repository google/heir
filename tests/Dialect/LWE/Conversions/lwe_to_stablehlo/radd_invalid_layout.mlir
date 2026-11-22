// RUN: heir-opt --lwe-to-stablehlo="r=4 c=8" --verify-diagnostics %s

!Z2147473409 = !mod_arith.int<2147473409 : i64>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#ring_16 = #polynomial.ring<coefficientType = !Z2147473409, polynomialModulus = <1 + x**16>>
#ct_space_16 = #lwe.ciphertext_space<ring = #ring_16, encryption_type = ckks>
!ct_16 = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = #ct_space_16, key = #key>

module {
  func.func @mismatched_degree_layout(%ct0: !ct_16, %ct1: !ct_16) -> !ct_16 {
    // expected-error @+1 {{degree layout (r=4, c=8) does not multiply to ring degree 16}}
    %sum = lwe.radd %ct0, %ct1 : (!ct_16, !ct_16) -> !ct_16
    return %sum : !ct_16
  }
}
