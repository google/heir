// RUN: heir-opt %s --lwe-to-polynomial --verify-diagnostics --split-input-file 2>&1

!Z4 = !mod_arith.int<4 : i32>
!Z3 = !mod_arith.int<3 : i32>
!Z5 = !mod_arith.int<5 : i32>
!rns_src_even = !rns.rns<!Z4, !Z3>
!rns_tgt_odd = !rns.rns<!Z5>
#ring_src_even = #polynomial.ring<coefficientType = !rns_src_even, polynomialModulus = <1 + x**8>>
#ring_tgt_odd = #polynomial.ring<coefficientType = !rns_tgt_odd, polynomialModulus = <1 + x**8>>
!ringelt_src_even = !lwe.lwe_ring_elt<ring = #ring_src_even>
!ringelt_tgt_odd = !lwe.lwe_ring_elt<ring = #ring_tgt_odd>

func.func @input_basis_contains_even_modulus(%arg0: !ringelt_src_even) -> !ringelt_tgt_odd {
  // expected-error@below {{basis conversion requires odd moduli, but input basis contains even modulus 4}}
  %0 = "lwe.convert_basis"(%arg0) {targetBasis = !rns_tgt_odd} : (!ringelt_src_even) -> !ringelt_tgt_odd
  return %0 : !ringelt_tgt_odd
}

// -----

!Z3 = !mod_arith.int<3 : i32>
!Z5 = !mod_arith.int<5 : i32>
!Z4 = !mod_arith.int<4 : i32>
!rns_src_odd = !rns.rns<!Z3, !Z5>
!rns_tgt_even = !rns.rns<!Z4>
#ring_src_odd = #polynomial.ring<coefficientType = !rns_src_odd, polynomialModulus = <1 + x**8>>
#ring_tgt_even = #polynomial.ring<coefficientType = !rns_tgt_even, polynomialModulus = <1 + x**8>>
!ringelt_src_odd = !lwe.lwe_ring_elt<ring = #ring_src_odd>
!ringelt_tgt_even = !lwe.lwe_ring_elt<ring = #ring_tgt_even>

func.func @target_basis_contains_even_modulus(%arg0: !ringelt_src_odd) -> !ringelt_tgt_even {
  // expected-error@below {{basis conversion requires odd moduli, but target basis contains even modulus 4}}
  %0 = "lwe.convert_basis"(%arg0) {targetBasis = !rns_tgt_even} : (!ringelt_src_odd) -> !ringelt_tgt_even
  return %0 : !ringelt_tgt_even
}
