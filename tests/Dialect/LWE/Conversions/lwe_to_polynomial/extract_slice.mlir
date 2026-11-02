// RUN: heir-opt --lwe-to-polynomial %s | FileCheck %s

#poly_mod = #polynomial.int_polynomial<1 + x**4>
!coeff0 = !mod_arith.int<17 : i32>
!coeff1 = !mod_arith.int<97 : i32>
!rns2 = !rns.rns<!coeff0, !coeff1>
!rns1 = !rns.rns<!coeff1>
#ring = #polynomial.ring<coefficientType=!rns2, polynomialModulus=#poly_mod>
#slice_ring = #polynomial.ring<coefficientType=!rns1, polynomialModulus=#poly_mod>
!input_ringelt = !lwe.lwe_ring_elt<ring=#ring>
!slice_ringelt = !lwe.lwe_ring_elt<ring=#slice_ring>

// CHECK: @extract_slice
func.func @extract_slice(%x: !input_ringelt) -> !slice_ringelt {
  // CHECK: polynomial.apply_coefficientwise
  // CHECK: rns.extract_slice {{.*}} <start = 1, size = 1>
  // CHECK: polynomial.yield
  %0 = lwe.extract_slice %x <start = 1, size = 1> : !input_ringelt -> !slice_ringelt
  return %0 : !slice_ringelt
}
