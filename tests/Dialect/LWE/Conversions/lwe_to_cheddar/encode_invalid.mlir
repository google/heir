// RUN: heir-opt --lwe-to-cheddar --verify-diagnostics %s

#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #encoding>>

module attributes {
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<
    logN = 16,
    Q = [36028797019488257, 1099512938497],
    P = [2305843009211596801],
    logDefaultScale = 40
  >
} {
  func.func @missing_level(%input: tensor<4xf64>) -> !pt {
    // expected-error @below {{failed to legalize operation 'lwe.rlwe_encode'}}
    // expected-error @below {{cannot lower to cheddar.encode without an explicit level}}
    %0 = lwe.rlwe_encode %input <
      encoding = #encoding,
      ring = #ring_f64
    > : tensor<4xf64> -> !pt
    return %0 : !pt
  }
}
