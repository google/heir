// RUN: heir-opt %s

// This test exists to prevent regressions of
// https://github.com/google/heir/pull/2671#issuecomment-3975431340.
// At one point, the issue was fixed in Polynomial (because tests were failing),
// but no tests were covering the corresponding LWE op. This test exercises the
// shape-inference path for `lwe.extract_slice`.

#poly_mod = #polynomial.int_polynomial<1 + x**4>
!coeff0 = !mod_arith.int<17 : i32>
!coeff1 = !mod_arith.int<97 : i32>
!rns2 = !rns.rns<!coeff0, !coeff1>
!rns1 = !rns.rns<!coeff0>
#ring = #polynomial.ring<coefficientType=!rns2, polynomialModulus=#poly_mod>
#slice_ring = #polynomial.ring<coefficientType=!rns1, polynomialModulus=#poly_mod>
!input_ringelt = !lwe.lwe_ring_elt<ring=#ring>
!slice_ringelt = !lwe.lwe_ring_elt<ring=#slice_ring>

module {
  func.func @preserve_extract_slice_shape(%x: !input_ringelt) {
    %0 = lwe.extract_slice %x {start = 0 : index, size = 1 : index}
        : !input_ringelt -> !slice_ringelt
    return
  }
}
