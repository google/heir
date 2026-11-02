// RUN: heir-opt %s

#poly_mod = #polynomial.int_polynomial<1 + x**4>
!coeff0 = !mod_arith.int<17 : i32>
!coeff1 = !mod_arith.int<97 : i32>
!rns2 = !rns.rns<!coeff0, !coeff1>
!rns1 = !rns.rns<!coeff0>
#ring = #polynomial.ring<coefficientType=!rns2, polynomialModulus=#poly_mod>
#slice_ring = #polynomial.ring<coefficientType=!rns1, polynomialModulus=#poly_mod>
!input_poly = !polynomial.polynomial<ring=#ring, form=eval>
!slice_poly = !polynomial.polynomial<ring=#slice_ring, form=eval>

module {
  func.func @preserve_extract_slice_form(
      %poly: !input_poly) {
    %0 = polynomial.apply_coefficientwise (%poly : !input_poly) {
    ^body(%coeff: !rns2, %degree: index):
      %1 = rns.extract_slice %coeff <start = 0, size = 1> : !rns2 -> !rns1
      polynomial.yield %1 : !rns1
    } -> !slice_poly
    return
  }
}
