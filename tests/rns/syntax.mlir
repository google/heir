// RUN: heir-opt --verify-diagnostics %s

#ideal = #polynomial.int_polynomial<1 + x**1024>
#ideal_2 = #polynomial.int_polynomial<1 + x**2048>

// Below we use random 32-bit primes
#ring_1 = #polynomial.ring<coefficientType = i32, coefficientModulus = 3721063133 : i32, polynomialModulus=#ideal>
#ring_2 = #polynomial.ring<coefficientType = i32, coefficientModulus = 2737228591 : i32, polynomialModulus=#ideal>
#ring_3 = #polynomial.ring<coefficientType = i32, coefficientModulus = 3180146689 : i32, polynomialModulus=#ideal>
#ring_bad = #polynomial.ring<coefficientType = i32, coefficientModulus = 3180146689 : i32, polynomialModulus=#ideal_2>

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
