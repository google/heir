// RUN: heir-opt -canonicalize %s | FileCheck %s --check-prefix=CANON
// RUN: heir-opt --attach-ntt-roots -canonicalize %s | FileCheck %s --check-prefix=NTT

#ideal = #polynomial.int_polynomial<1 + x**4>
!i32_poly_ty = !polynomial.polynomial<ring=<coefficientType=i32, polynomialModulus=#ideal>>

!Z17_i32 = !mod_arith.int<17 : i32>
!mod_poly_ty = !polynomial.polynomial<ring=<coefficientType=!Z17_i32, polynomialModulus=#ideal>>
!mod_poly_ev_ty = !polynomial.polynomial<ring=<coefficientType=!Z17_i32, polynomialModulus=#ideal>, form=eval>

// CANON: @test_fold_monomial_arith_coefficient
func.func @test_fold_monomial_arith_coefficient() -> !i32_poly_ty {
  // CANON-NOT: polynomial.monomial
  // CANON: %[[CST:.*]] = polynomial.constant int<-5x**3> : [[TY:![a-zA-Z0-9_]+]]
  // CANON: return %[[CST]] : [[TY]]
  %coeff = arith.constant -5 : i32
  %degree = arith.constant 3 : index
  %monomial = polynomial.monomial %coeff, %degree : (i32, index) -> !i32_poly_ty
  return %monomial : !i32_poly_ty
}

// CANON: @test_fold_monomial_mod_arith_coefficient
func.func @test_fold_monomial_mod_arith_coefficient() -> !mod_poly_ty {
  // CANON-NOT: polynomial.monomial
  // CANON: %[[CST:.*]] = polynomial.constant int<5x**2> : [[TY:![a-zA-Z0-9_]+]]
  // CANON: return %[[CST]] : [[TY]]
  %coeff = mod_arith.constant 5 : !Z17_i32
  %degree = arith.constant 2 : index
  %monomial = polynomial.monomial %coeff, %degree : (!Z17_i32, index) -> !mod_poly_ty
  return %monomial : !mod_poly_ty
}

// CANON: @test_fold_monomial_degree_zero
func.func @test_fold_monomial_degree_zero() -> !i32_poly_ty {
  // CANON-NOT: polynomial.monomial
  // CANON: %[[CST:.*]] = polynomial.constant int<7> : [[TY:![a-zA-Z0-9_]+]]
  // CANON: return %[[CST]] : [[TY]]
  %coeff = arith.constant 7 : i32
  %degree = arith.constant 0 : index
  %monomial = polynomial.monomial %coeff, %degree : (i32, index) -> !i32_poly_ty
  return %monomial : !i32_poly_ty
}

// CANON: @test_fold_monomial_zero_coefficient
func.func @test_fold_monomial_zero_coefficient() -> !i32_poly_ty {
  // CANON-NOT: polynomial.monomial
  // CANON: %[[CST:.*]] = polynomial.constant int<> : [[TY:![a-zA-Z0-9_]+]]
  // CANON: return %[[CST]] : [[TY]]
  %coeff = arith.constant 0 : i32
  %degree = arith.constant 2 : index
  %monomial = polynomial.monomial %coeff, %degree : (i32, index) -> !i32_poly_ty
  return %monomial : !i32_poly_ty
}

// CANON: @test_no_fold_monomial_dynamic_coefficient
func.func @test_no_fold_monomial_dynamic_coefficient(%coeff: i32) -> !i32_poly_ty {
  // CANON: %[[DEGREE:.*]] = arith.constant 1 : index
  // CANON: %[[MONOMIAL:.*]] = polynomial.monomial %{{.*}}, %[[DEGREE]] : (i32, index) -> [[TY:![a-zA-Z0-9_]+]]
  // CANON: return %[[MONOMIAL]] : [[TY]]
  %degree = arith.constant 1 : index
  %monomial = polynomial.monomial %coeff, %degree : (i32, index) -> !i32_poly_ty
  return %monomial : !i32_poly_ty
}

// CANON: @test_no_fold_monomial_dynamic_degree
func.func @test_no_fold_monomial_dynamic_degree(%degree: index) -> !i32_poly_ty {
  // CANON: %[[COEFF:.*]] = arith.constant 1 : i32
  // CANON: %[[MONOMIAL:.*]] = polynomial.monomial %[[COEFF]], %{{.*}} : (i32, index) -> [[TY:![a-zA-Z0-9_]+]]
  // CANON: return %[[MONOMIAL]] : [[TY]]
  %coeff = arith.constant 1 : i32
  %monomial = polynomial.monomial %coeff, %degree : (i32, index) -> !i32_poly_ty
  return %monomial : !i32_poly_ty
}

// NTT: @test_fold_ntt_of_constant_monomial
func.func @test_fold_ntt_of_constant_monomial() -> !mod_poly_ev_ty {
  // NTT-NOT: polynomial.monomial
  // NTT-NOT: polynomial.ntt
  // NTT: %[[CST:.*]] = polynomial.constant int<{{.*}}> : [[TY:![a-zA-Z0-9_]+]]
  // NTT: return %[[CST]] : [[TY]]
  %coeff = mod_arith.constant 5 : !Z17_i32
  %degree = arith.constant 1 : index
  %monomial = polynomial.monomial %coeff, %degree : (!Z17_i32, index) -> !mod_poly_ty
  %ntt = polynomial.ntt %monomial : !mod_poly_ty
  return %ntt : !mod_poly_ev_ty
}
