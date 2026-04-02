// RUN: heir-opt %s --rns-lower-convert-basis --verify-diagnostics --split-input-file 2>&1

!Z4 = !mod_arith.int<4 : i32>
!Z3 = !mod_arith.int<3 : i32>
!Z5 = !mod_arith.int<5 : i32>
!rns_src_even = !rns.rns<!Z4, !Z3>
!rns_tgt_odd = !rns.rns<!Z5>

func.func @input_basis_contains_even_modulus(%arg0: !rns_src_even) -> !rns_tgt_odd {
  // expected-error@below {{basis conversion requires odd moduli, but input basis contains even modulus 4}}
  %0 = "rns.convert_basis"(%arg0) {targetBasis = !rns_tgt_odd} : (!rns_src_even) -> !rns_tgt_odd
  return %0 : !rns_tgt_odd
}

// -----

!Z3 = !mod_arith.int<3 : i32>
!Z5 = !mod_arith.int<5 : i32>
!Z4 = !mod_arith.int<4 : i32>
!rns_src_odd = !rns.rns<!Z3, !Z5>
!rns_tgt_even = !rns.rns<!Z4>

func.func @target_basis_contains_even_modulus(%arg0: !rns_src_odd) -> !rns_tgt_even {
  // expected-error@below {{basis conversion requires odd moduli, but target basis contains even modulus 4}}
  %0 = "rns.convert_basis"(%arg0) {targetBasis = !rns_tgt_even} : (!rns_src_odd) -> !rns_tgt_even
  return %0 : !rns_tgt_even
}
