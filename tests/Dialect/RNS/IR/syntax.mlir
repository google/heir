// RUN: heir-opt --verify-diagnostics --split-input-file %s

#ideal = #polynomial.int_polynomial<1 + x**1024>
#ideal_2 = #polynomial.int_polynomial<1 + x**2048>

// Below we use random 32-bit primes
#ring_1 = #polynomial.ring<coefficientType = i32, coefficientModulus = 3721063133 : i64, polynomialModulus=#ideal>
#ring_2 = #polynomial.ring<coefficientType = i32, coefficientModulus = 2737228591 : i64, polynomialModulus=#ideal>
#ring_3 = #polynomial.ring<coefficientType = i32, coefficientModulus = 3180146689 : i64, polynomialModulus=#ideal>
#ring_bad = #polynomial.ring<coefficientType = i32, coefficientModulus = 3180146689 : i64, polynomialModulus=#ideal_2>

!poly_ty_1 = !polynomial.polynomial<ring=#ring_1>
!poly_ty_2 = !polynomial.polynomial<ring=#ring_2>
!poly_ty_3 = !polynomial.polynomial<ring=#ring_3>

!ty = !rns.rns<!poly_ty_1, !poly_ty_2, !poly_ty_3>

func.func @test_syntax(%arg0: !ty) -> !ty {
  return %arg0 : !ty
}

!poly_ty_bad = !polynomial.polynomial<ring=#ring_bad>
// expected-error@+1 {{RNS type has incompatible basis types}}
!ty_bad = !rns.rns<!poly_ty_1, !poly_ty_2, !poly_ty_bad>

// -----
// mod arith

!Zp1 = !mod_arith.int<3721063133 : i64>
!Zp2 = !mod_arith.int<2737228591 : i64>
!Zp3 = !mod_arith.int<3180146689 : i64>

!ty_modarith = !rns.rns<!Zp1, !Zp2, !Zp3>

func.func @test_syntax_modarith(%arg0: !ty_modarith) -> !ty_modarith {
  return %arg0 : !ty_modarith
}

// expected-error@+1 {{RNS type has incompatible basis types}}
!ty_modarith_bad = !rns.rns<!Zp1, !Zp2, !Zp1>

// -----

!Zp1 = !mod_arith.int<3721063133 : i64>
!Zp2 = !mod_arith.int<65537 : i64>
!Zp2_i32 = !mod_arith.int<65537 : i32>

!ty_modarith = !rns.rns<!Zp1, !Zp2>

func.func @test_syntax_modarith(%arg0: !ty_modarith) -> !ty_modarith {
  return %arg0 : !ty_modarith
}

// expected-error@+1 {{RNS type has incompatible basis types}}
!ty_modarith_bad = !rns.rns<!Zp1, !Zp2_i32>

// -----

// expected-error@+1 {{does not have RNSBasisTypeInterface}}
!ty_int_bad = !rns.rns<i32, i64>
