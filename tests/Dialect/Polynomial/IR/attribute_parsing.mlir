// RUN: heir-opt %s | FileCheck %s

#ideal = #polynomial.int_polynomial<1 + x**2>
!rns_basis_0 = !mod_arith.int<17 : i32>
!rns_basis_1 = !mod_arith.int<19 : i32>
!rns_ty = !rns.rns<!rns_basis_0, !rns_basis_1>
!rns_poly_ty = !polynomial.polynomial<ring=<coefficientType=!rns_ty, polynomialModulus=#ideal>>

#rns_poly = #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
#rns_poly_coeff = #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, coeff> : !rns_poly_ty
#rns_poly_eval = #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ty

// CHECK: func @test_rns_poly
func.func @test_rns_poly() -> !rns_poly_ty {
  // CHECK: polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}1, 2{{\]}}, {{\[}}3, 4{{\]\]}}> : tensor<2x2xi32>> : {{![a-zA-Z0-9_]+}}> : {{![a-zA-Z0-9_]+}}
  %0 = polynomial.constant #rns_poly
  return %0 : !rns_poly_ty
}

// CHECK: func @test_rns_poly_coeff
func.func @test_rns_poly_coeff() -> !rns_poly_ty {
  // CHECK: polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}1, 2{{\]}}, {{\[}}3, 4{{\]\]}}> : tensor<2x2xi32>> : {{![a-zA-Z0-9_]+}}> : {{![a-zA-Z0-9_]+}}
  %0 = polynomial.constant #rns_poly_coeff
  return %0 : !rns_poly_ty
}

// CHECK: func @test_rns_poly_eval
func.func @test_rns_poly_eval() -> !rns_poly_ty {
  // CHECK: polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}1, 2{{\]}}, {{\[}}3, 4{{\]\]}}> : tensor<2x2xi32>, eval> : {{![a-zA-Z0-9_]+}}> : {{![a-zA-Z0-9_]+}}
  %0 = polynomial.constant #rns_poly_eval
  return %0 : !rns_poly_ty
}
