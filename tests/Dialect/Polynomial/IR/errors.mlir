// RUN: heir-opt %s --verify-diagnostics --split-input-file

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>
#one_plus_x_squared = #polynomial.int_polynomial<1 + x**2>

func.func @test_to_from_tensor(%p0 : !polynomial.polynomial<ring=#ring>) {
  %c0 = arith.constant 0 : index
  %two = arith.constant 2 : i32
  %coeffs = tensor.from_elements %two, %two : tensor<2xi32>
  // expected-error@below {{'polynomial.from_tensor' op polynomial coefficient type '!mod_arith.int<2837465 : i32>' does not match scalar type 'i32'}}
  %poly = polynomial.from_tensor %coeffs : tensor<2xi32> -> !polynomial.polynomial<ring=#ring>
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>
#one_plus_x_squared = #polynomial.int_polynomial<1 + x**2>

func.func @test_degree(%p0 : !polynomial.polynomial<ring=#ring>) {
  // expected-error@below {{'polynomial.leading_term' op polynomial coefficient type '!mod_arith.int<2837465 : i32>' does not match scalar type 'i32'}}
  %0, %1 = polynomial.leading_term %p0 : !polynomial.polynomial<ring=#ring> -> (index, i32)
  return
}

// -----

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>
#one_plus_x_squared = #polynomial.int_polynomial<1 + x**2>

func.func @test_monomial() {
  %deg = arith.constant 1023 : index
  %five = arith.constant 5 : i32
  // expected-error@below {{'polynomial.monomial' op polynomial coefficient type '!mod_arith.int<2837465 : i32>' does not match scalar type 'i32'}}
  %0 = polynomial.monomial %five, %deg : (i32, index) -> !polynomial.polynomial<ring=#ring>
  return
}

// -----

#invalid_ideal = #polynomial.int_polynomial<-1 + x**2>
!rns_basis_0 = !mod_arith.int<17 : i32>
!rns_basis_1 = !mod_arith.int<19 : i32>
!rns_ty = !rns.rns<!rns_basis_0, !rns_basis_1>
!rns_poly_ty = !polynomial.polynomial<ring=<coefficientType=!rns_ty, polynomialModulus=#invalid_ideal>>

// expected-error@below {{RNS polynomial attributes require a polynomialModulus of the form x^N + 1, but got #polynomial.int_polynomial<-1 + x**2>}}
#rns_poly = #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty

func.func @test_invalid_modulus() -> !rns_poly_ty {
  %0 = polynomial.constant #rns_poly
  return %0 : !rns_poly_ty
}
