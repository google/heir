// RUN: heir-opt --polynomial-to-mod-arith --verify-diagnostics %s

!rns_basis_0 = !mod_arith.int<17 : i32>
!rns_basis_1 = !mod_arith.int<19 : i32>
!rns_ty = !rns.rns<!rns_basis_0, !rns_basis_1>

#poly_mod = #polynomial.int_polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType=!rns_ty, polynomialModulus=#poly_mod>
!rns_poly_ty = !polynomial.polynomial<ring=#ring>

#rns_poly = #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty

func.func @test_rns_poly_failure() -> !rns_poly_ty {
  // expected-warning@+2 {{Native lowering for RNSPolynomialAttr is not implemented yet}}
  // expected-error@+1 {{failed to legalize operation}}
  %0 = polynomial.constant #rns_poly
  return %0 : !rns_poly_ty
}
