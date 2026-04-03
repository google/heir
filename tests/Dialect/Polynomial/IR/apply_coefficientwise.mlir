// RUN: heir-opt %s --verify-diagnostics --split-input-file

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>

func.func @test_apply_coefficientwise(%p0 : !polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring> {
  %1 = polynomial.apply_coefficientwise (%p0 : !polynomial.polynomial<ring=#ring>) {
  ^body(%coeff: !coeff_ty, %degree: index):
    %2 = mod_arith.add %coeff, %coeff : !coeff_ty
    polynomial.yield %2 : !coeff_ty
  } -> !polynomial.polynomial<ring=#ring>
  return %1 : !polynomial.polynomial<ring=#ring>
}

// -----

// The example from the documentation string
#ring1 = #polynomial.ring<coefficientType=i32>
#ring2 = #polynomial.ring<coefficientType=f32>
!poly_ty_1 = !polynomial.polynomial<ring=#ring1>
!poly_ty_2 = !polynomial.polynomial<ring=#ring2>

func.func @test_docs_example(%0 : !poly_ty_1) -> !poly_ty_2 {
  %divisor = arith.constant 3.4 : f32
  %1 = polynomial.apply_coefficientwise(%0 : !poly_ty_1) {
  ^body(%coeff: i32, %degree: index):
    %2 = arith.sitofp %coeff : i32 to f32
    %3 = arith.divf %2, %divisor : f32
    polynomial.yield %3 : f32
  } -> !poly_ty_2
  return %1 : !poly_ty_2
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
!rns_ty = !rns.rns<!mod_arith.int<2837465:i32>, !mod_arith.int<2837467:i32>>
#ring = #polynomial.ring<coefficientType=!rns_ty, polynomialModulus=#poly>

func.func @test_apply_coefficientwise_rns(%p0 : !polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring> {
  %1 = polynomial.apply_coefficientwise (%p0 : !polynomial.polynomial<ring=#ring>) {
  ^body(%coeff: !rns_ty, %degree: index):
    // Just yield the same RNS value
    polynomial.yield %coeff : !rns_ty
  } -> !polynomial.polynomial<ring=#ring>
  return %1 : !polynomial.polynomial<ring=#ring>
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>

func.func @test_invalid_num_args(%p0 : !polynomial.polynomial<ring=#ring>) {
  // expected-error@below {{'polynomial.apply_coefficientwise' op requires a body with 2 arguments: the coefficient and the degree}}
  %1 = polynomial.apply_coefficientwise (%p0 : !polynomial.polynomial<ring=#ring>) {
  ^body(%coeff: !coeff_ty):
    polynomial.yield %coeff : !coeff_ty
  } -> !polynomial.polynomial<ring=#ring>
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>

func.func @test_invalid_first_arg_type(%p0 : !polynomial.polynomial<ring=#ring>) {
  // expected-error@below {{expected first argument of the body to be of type '!mod_arith.int<2837465 : i32>', but found 'i32'}}
  %1 = polynomial.apply_coefficientwise (%p0 : !polynomial.polynomial<ring=#ring>) {
  ^body(%coeff: i32, %degree: index):
    %2 = arith.constant 0 : i32
    polynomial.yield %coeff : i32 // This would also error on yield if coeff was i32
  } -> !polynomial.polynomial<ring=#ring>
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>

func.func @test_invalid_second_arg_type(%p0 : !polynomial.polynomial<ring=#ring>) {
  // expected-error@below {{expected second argument of the body to be of type index, but found 'i32'}}
  %1 = polynomial.apply_coefficientwise (%p0 : !polynomial.polynomial<ring=#ring>) {
  ^body(%coeff: !coeff_ty, %degree: i32):
    polynomial.yield %coeff : !coeff_ty
  } -> !polynomial.polynomial<ring=#ring>
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>

func.func @test_invalid_yield_type(%p0 : !polynomial.polynomial<ring=#ring>) {
  // expected-error@below {{expected yield operand to be of type '!mod_arith.int<2837465 : i32>', but found 'i32'}}
  %1 = polynomial.apply_coefficientwise (%p0 : !polynomial.polynomial<ring=#ring>) {
  ^body(%coeff: !coeff_ty, %degree: index):
    %2 = arith.constant 0 : i32
    polynomial.yield %2 : i32
  } -> !polynomial.polynomial<ring=#ring>
  return
}
