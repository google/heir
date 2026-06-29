// RUN: heir-opt -canonicalize %s | FileCheck %s

#ideal = #polynomial.int_polynomial<1 + x**2>
!rns_basis_0 = !mod_arith.int<17 : i32>
!rns_basis_1 = !mod_arith.int<13 : i32>
!rns_ty = !rns.rns<!rns_basis_0, !rns_basis_1>
!rns_poly_ty = !polynomial.polynomial<ring=<coefficientType=!rns_ty, polynomialModulus=#ideal>>
!rns_poly_ev_ty = !polynomial.polynomial<ring=<coefficientType=!rns_ty, polynomialModulus=#ideal>, form=eval>

!rns_sliced_basis = !rns.rns<!rns_basis_1>
!rns_sliced_poly_ty = !polynomial.polynomial<ring=<coefficientType=!rns_sliced_basis, polynomialModulus=#ideal>>

#v0 = #mod_arith.value<4 : !rns_basis_0>
#v1 = #mod_arith.value<5 : !rns_basis_1>
#rns_root = #rns.value<[#v0, #v1]>
#root = #polynomial.primitive_root<value = #rns_root, degree = 4 : i32>

// CHECK: @test_fold_add
func.func @test_fold_add() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.add
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}2, 4{{\]}}, {{\[}}6, 8{{\]\]}}> : tensor<2x2xi32>> : [[TY1:![a-zA-Z0-9_]+]]> : [[TY1]]
  // CHECK: return %[[CST]] : [[TY1]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %2 = polynomial.add %0, %1 : !rns_poly_ty
  return %2 : !rns_poly_ty
}

// CHECK: @test_fold_add_with_reduction
func.func @test_fold_add_with_reduction() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.add
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}3, 0{{\]}}, {{\[}}4, 8{{\]\]}}> : tensor<2x2xi32>> : [[TY2:![a-zA-Z0-9_]+]]> : [[TY2]]
  // CHECK: return %[[CST]] : [[TY2]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[10, 9], [12, 11]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[10, 8], [5, 10]]> : tensor<2x2xi32>> : !rns_poly_ty
  %2 = polynomial.add %0, %1 : !rns_poly_ty
  return %2 : !rns_poly_ty
}

// CHECK: @test_fold_sub
func.func @test_fold_sub() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.sub
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<0> : tensor<2x2xi32>> : [[TY3:![a-zA-Z0-9_]+]]> : [[TY3]]
  // CHECK: return %[[CST]] : [[TY3]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %2 = polynomial.sub %0, %1 : !rns_poly_ty
  return %2 : !rns_poly_ty
}

// CHECK: @test_fold_mul_ntt
func.func @test_fold_mul_ntt() -> !rns_poly_ev_ty {
  // CHECK-NOT: polynomial.mul
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}1, 4{{\]}}, {{\[}}9, 3{{\]\]}}> : tensor<2x2xi32>, eval> : [[TY4:![a-zA-Z0-9_]+]]> : [[TY4]]
  // CHECK: return %[[CST]] : [[TY4]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ev_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ev_ty
  %2 = polynomial.mul %0, %1 : !rns_poly_ev_ty
  return %2 : !rns_poly_ev_ty
}

// CHECK: @test_fold_ntt
func.func @test_fold_ntt() -> !rns_poly_ev_ty {
  // CHECK-NOT: polynomial.ntt
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}9, 10{{\]}}, {{\[}}10, 9{{\]\]}}> : tensor<2x2xi32>, eval> : [[TY5:![a-zA-Z0-9_]+]]> : [[TY5]]
  // CHECK: return %[[CST]] : [[TY5]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.ntt %0 {root = #root} : !rns_poly_ty
  return %1 : !rns_poly_ev_ty
}

// CHECK: @test_fold_intt
func.func @test_fold_intt() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.intt
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}10, 2{{\]}}, {{\[}}10, 9{{\]\]}}> : tensor<2x2xi32>> : [[TY6:![a-zA-Z0-9_]+]]> : [[TY6]]
  // CHECK: return %[[CST]] : [[TY6]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ev_ty
  %1 = polynomial.intt %0 {root = #root} : !rns_poly_ev_ty
  return %1 : !rns_poly_ty
}

// CHECK: @test_fold_extract_slice
func.func @test_fold_extract_slice() -> !rns_sliced_poly_ty {
  // CHECK-NOT: polynomial.extract_slice
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}3, 4{{\]\]}}> : tensor<1x2xi32>> : [[TY7:![a-zA-Z0-9_]+]]> : [[TY7]]
  // CHECK: return %[[CST]] : [[TY7]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.extract_slice %0 {start = 1 : index, size = 1 : index} : !rns_poly_ty -> !rns_sliced_poly_ty
  return %1 : !rns_sliced_poly_ty
}

// CHECK: @test_no_fold_ntt_missing_root
func.func @test_no_fold_ntt_missing_root() -> !rns_poly_ev_ty {
  // CHECK: %[[CST:.*]] = polynomial.constant
  // CHECK: %[[NTT:.*]] = polynomial.ntt %[[CST]] : [[TY8:![a-zA-Z0-9_]+]]
  // CHECK: return %[[NTT]] : [[RET_TY8:![a-zA-Z0-9_]+]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.ntt %0 : !rns_poly_ty
  return %1 : !rns_poly_ev_ty
}

// CHECK: @test_no_fold_intt_missing_root
func.func @test_no_fold_intt_missing_root() -> !rns_poly_ty {
  // CHECK: %[[CST:.*]] = polynomial.constant
  // CHECK: %[[INTT:.*]] = polynomial.intt %[[CST]] : [[TY9:![a-zA-Z0-9_]+]]
  // CHECK: return %[[INTT]] : [[RET_TY9:![a-zA-Z0-9_]+]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ev_ty
  %1 = polynomial.intt %0 : !rns_poly_ev_ty
  return %1 : !rns_poly_ty
}
