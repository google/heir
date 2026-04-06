// RUN: heir-opt %s --verify-diagnostics --split-input-file

#poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
!coeff_poly = !polynomial.polynomial<ring=#ring, form=coeff>
!eval_poly = !polynomial.polynomial<ring=#ring, form=eval>

func.func @test_add_mismatched_forms(%p0 : !coeff_poly, %p1 : !eval_poly) {
  // expected-error@below {{'polynomial.add' op requires all polynomial operands and results to have the same form}}
  %0 = "polynomial.add"(%p0, %p1) : (!coeff_poly, !eval_poly) -> !coeff_poly
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
!coeff_poly = !polynomial.polynomial<ring=#ring, form=coeff>
!eval_poly = !polynomial.polynomial<ring=#ring, form=eval>

func.func @test_add_tensor_mismatched_forms(%p0 : tensor<2x!coeff_poly>, %p1 : tensor<2x!eval_poly>) {
  // expected-error@below {{'polynomial.add' op requires all polynomial operands and results to have the same form}}
  %0 = "polynomial.add"(%p0, %p1) : (tensor<2x!coeff_poly>, tensor<2x!eval_poly>) -> tensor<2x!coeff_poly>
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
#ring32 = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
#ring64 = #polynomial.ring<coefficientType=i64, polynomialModulus=#poly>
!eval_poly64 = !polynomial.polynomial<ring=#ring64, form=eval>
!coeff_poly32 = !polynomial.polynomial<ring=#ring32, form=coeff>

func.func @test_mod_switch_mismatched_forms(%p0 : !eval_poly64) {
  // expected-error@below {{'polynomial.mod_switch' op requires all polynomial operands and results to have the same form}}
  %0 = polynomial.mod_switch %p0 : !eval_poly64 to !coeff_poly32
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
!eval_poly = !polynomial.polynomial<ring=#ring, form=eval>
!coeff_poly = !polynomial.polynomial<ring=#ring, form=coeff>

func.func @test_extract_slice_mismatched_forms(%p0 : !eval_poly) {
  // expected-error@below {{'polynomial.extract_slice' op requires all polynomial operands and results to have the same form}}
  %0 = "polynomial.extract_slice"(%p0) {start = 0 : index, size = 1 : index} : (!eval_poly) -> !coeff_poly
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
!coeff_poly = !polynomial.polynomial<ring=#ring, form=coeff>
!eval_poly = !polynomial.polynomial<ring=#ring, form=eval>

func.func @test_mul_mismatched_forms(%p0 : !coeff_poly, %p1 : !eval_poly) {
  // expected-error@below {{'polynomial.mul' op requires all polynomial operands and results to have the same form}}
  %0 = "polynomial.mul"(%p0, %p1) : (!coeff_poly, !eval_poly) -> !coeff_poly
  return
}
