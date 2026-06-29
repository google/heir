// RUN: heir-opt -canonicalize %s | FileCheck %s

#ideal = #polynomial.int_polynomial<1 + x**2>
#ideal4 = #polynomial.int_polynomial<1 + x**4>
!rns_basis_0 = !mod_arith.int<17 : i32>
!rns_basis_1 = !mod_arith.int<13 : i32>
!rns_ty = !rns.rns<!rns_basis_0, !rns_basis_1>
!rns_poly_ty = !polynomial.polynomial<ring=<coefficientType=!rns_ty, polynomialModulus=#ideal>>
!rns_poly_ev_ty = !polynomial.polynomial<ring=<coefficientType=!rns_ty, polynomialModulus=#ideal>, form=eval>

!mod_poly_ty = !polynomial.polynomial<ring=<coefficientType=!rns_basis_0, polynomialModulus=#ideal>>
!mod_poly_ev_ty = !polynomial.polynomial<ring=<coefficientType=!rns_basis_0, polynomialModulus=#ideal>, form=eval>
!mod_int_poly_ty = !polynomial.polynomial<ring=<coefficientType=!rns_basis_0, polynomialModulus=#ideal4>>
!mod_int_poly_ev_ty = !polynomial.polynomial<ring=<coefficientType=!rns_basis_0, polynomialModulus=#ideal4>, form=eval>
!int_poly_ty = !polynomial.polynomial<ring=<coefficientType=i32, polynomialModulus=#ideal4>>
!int_poly_ev_ty = !polynomial.polynomial<ring=<coefficientType=i32, polynomialModulus=#ideal>, form=eval>
!float_poly_ty = !polynomial.polynomial<ring=<coefficientType=f32, polynomialModulus=#ideal4>>

!rns_sliced_basis = !rns.rns<!rns_basis_1>
!rns_sliced_poly_ty = !polynomial.polynomial<ring=<coefficientType=!rns_sliced_basis, polynomialModulus=#ideal>>

#v0 = #mod_arith.value<4 : !rns_basis_0>
#v1 = #mod_arith.value<5 : !rns_basis_1>
#rns_root = #rns.value<[#v0, #v1]>
#root = #polynomial.primitive_root<value = #rns_root, degree = 4 : i32>
#mod_root = #polynomial.primitive_root<value = #v0, degree = 4 : i32>

// CHECK: @test_fold_rns_add
func.func @test_fold_rns_add() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.add
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}2, 4{{\]}}, {{\[}}6, 8{{\]\]}}> : tensor<2x2xi32>> : [[TY1:![a-zA-Z0-9_]+]]> : [[TY1]]
  // CHECK: return %[[CST]] : [[TY1]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %2 = polynomial.add %0, %1 : !rns_poly_ty
  return %2 : !rns_poly_ty
}

// CHECK: @test_fold_rns_add_with_reduction
func.func @test_fold_rns_add_with_reduction() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.add
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}3, 0{{\]}}, {{\[}}4, 8{{\]\]}}> : tensor<2x2xi32>> : [[TY2:![a-zA-Z0-9_]+]]> : [[TY2]]
  // CHECK: return %[[CST]] : [[TY2]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[10, 9], [12, 11]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[10, 8], [5, 10]]> : tensor<2x2xi32>> : !rns_poly_ty
  %2 = polynomial.add %0, %1 : !rns_poly_ty
  return %2 : !rns_poly_ty
}

// CHECK: @test_fold_rns_sub
func.func @test_fold_rns_sub() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.sub
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<0> : tensor<2x2xi32>> : [[TY3:![a-zA-Z0-9_]+]]> : [[TY3]]
  // CHECK: return %[[CST]] : [[TY3]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %2 = polynomial.sub %0, %1 : !rns_poly_ty
  return %2 : !rns_poly_ty
}

// CHECK: @test_fold_rns_mul_eval
func.func @test_fold_rns_mul_eval() -> !rns_poly_ev_ty {
  // CHECK-NOT: polynomial.mul
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}1, 4{{\]}}, {{\[}}9, 3{{\]\]}}> : tensor<2x2xi32>, eval> : [[TY4:![a-zA-Z0-9_]+]]> : [[TY4]]
  // CHECK: return %[[CST]] : [[TY4]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ev_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ev_ty
  %2 = polynomial.mul %0, %1 : !rns_poly_ev_ty
  return %2 : !rns_poly_ev_ty
}

// CHECK: @test_fold_rns_mul_coeff
func.func @test_fold_rns_mul_coeff() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.mul
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}0, 2{{\]}}, {{\[}}0, 2{{\]\]}}> : tensor<2x2xi32>> : [[TY_RNS_MUL_COEFF:![a-zA-Z0-9_]+]]> : [[TY_RNS_MUL_COEFF]]
  // CHECK: return %[[CST]] : [[TY_RNS_MUL_COEFF]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 1], [1, 1]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 1], [1, 1]]> : tensor<2x2xi32>> : !rns_poly_ty
  %2 = polynomial.mul %0, %1 : !rns_poly_ty
  return %2 : !rns_poly_ty
}

// CHECK: @test_fold_rns_monomial
func.func @test_fold_rns_monomial() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.monomial
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}0, 4{{\]}}, {{\[}}0, 5{{\]\]}}> : tensor<2x2xi32>> : [[TY_RNS_MONOMIAL:![a-zA-Z0-9_]+]]> : [[TY_RNS_MONOMIAL]]
  // CHECK: return %[[CST]] : [[TY_RNS_MONOMIAL]]
  %degree = arith.constant 1 : index
  %coeff = rns.constant <[#mod_arith.value<4 : !rns_basis_0>, #mod_arith.value<5 : !rns_basis_1>]> : !rns_ty
  %0 = polynomial.monomial %coeff, %degree : (!rns_ty, index) -> !rns_poly_ty
  return %0 : !rns_poly_ty
}

// CHECK: @test_fold_rns_monic_monomial_mul
func.func @test_fold_rns_monic_monomial_mul() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.monic_monomial_mul
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}13, 0{{\]}}, {{\[}}8, 0{{\]\]}}> : tensor<2x2xi32>> : [[TY_RNS_MONIC_MONOMIAL_MUL:![a-zA-Z0-9_]+]]> : [[TY_RNS_MONIC_MONOMIAL_MUL]]
  // CHECK: return %[[CST]] : [[TY_RNS_MONIC_MONOMIAL_MUL]]
  %degree = arith.constant 1 : index
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[0, 4], [0, 5]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.monic_monomial_mul %0, %degree : (!rns_poly_ty, index) -> !rns_poly_ty
  return %1 : !rns_poly_ty
}

// CHECK: @test_fold_rns_mul_scalar
func.func @test_fold_rns_mul_scalar() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.mul_scalar
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}3, 6{{\]}}, {{\[}}2, 4{{\]\]}}> : tensor<2x2xi32>> : [[TY_RNS_MUL_SCALAR:![a-zA-Z0-9_]+]]> : [[TY_RNS_MUL_SCALAR]]
  // CHECK: return %[[CST]] : [[TY_RNS_MUL_SCALAR]]
  %scalar = rns.constant <[#mod_arith.value<3 : !rns_basis_0>, #mod_arith.value<7 : !rns_basis_1>]> : !rns_ty
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [4, 8]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.mul_scalar %0, %scalar : !rns_poly_ty, !rns_ty
  return %1 : !rns_poly_ty
}

// CHECK: @test_fold_rns_ntt
func.func @test_fold_rns_ntt() -> !rns_poly_ev_ty {
  // CHECK-NOT: polynomial.ntt
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}9, 10{{\]}}, {{\[}}10, 9{{\]\]}}> : tensor<2x2xi32>, eval> : [[TY5:![a-zA-Z0-9_]+]]> : [[TY5]]
  // CHECK: return %[[CST]] : [[TY5]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.ntt %0 {root = #root} : !rns_poly_ty
  return %1 : !rns_poly_ev_ty
}

// CHECK: @test_fold_rns_intt
func.func @test_fold_rns_intt() -> !rns_poly_ty {
  // CHECK-NOT: polynomial.intt
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}10, 2{{\]}}, {{\[}}10, 9{{\]\]}}> : tensor<2x2xi32>> : [[TY6:![a-zA-Z0-9_]+]]> : [[TY6]]
  // CHECK: return %[[CST]] : [[TY6]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ev_ty
  %1 = polynomial.intt %0 {root = #root} : !rns_poly_ev_ty
  return %1 : !rns_poly_ty
}


// CHECK: @test_fold_rns_extract_slice
func.func @test_fold_rns_extract_slice() -> !rns_sliced_poly_ty {
  // CHECK-NOT: polynomial.extract_slice
  // CHECK: %[[CST:.*]] = polynomial.constant #polynomial<rns_polynomial<dense<{{\[\[}}3, 4{{\]\]}}> : tensor<1x2xi32>> : [[TY7:![a-zA-Z0-9_]+]]> : [[TY7]]
  // CHECK: return %[[CST]] : [[TY7]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.extract_slice %0 {start = 1 : index, size = 1 : index} : !rns_poly_ty -> !rns_sliced_poly_ty
  return %1 : !rns_sliced_poly_ty
}

// CHECK: @test_no_fold_rns_ntt_missing_root
func.func @test_no_fold_rns_ntt_missing_root() -> !rns_poly_ev_ty {
  // CHECK: %[[CST:.*]] = polynomial.constant
  // CHECK: %[[NTT:.*]] = polynomial.ntt %[[CST]] : [[TY8:![a-zA-Z0-9_]+]]
  // CHECK: return %[[NTT]] : [[RET_TY8:![a-zA-Z0-9_]+]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !rns_poly_ty
  %1 = polynomial.ntt %0 : !rns_poly_ty
  return %1 : !rns_poly_ev_ty
}

// CHECK: @test_no_fold_rns_intt_missing_root
func.func @test_no_fold_rns_intt_missing_root() -> !rns_poly_ty {
  // CHECK: %[[CST:.*]] = polynomial.constant
  // CHECK: %[[INTT:.*]] = polynomial.intt %[[CST]] : [[TY9:![a-zA-Z0-9_]+]]
  // CHECK: return %[[INTT]] : [[RET_TY9:![a-zA-Z0-9_]+]]
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>, eval> : !rns_poly_ev_ty
  %1 = polynomial.intt %0 : !rns_poly_ev_ty
  return %1 : !rns_poly_ty
}





// CHECK: @test_fold_mod_arith_add_with_reduction
func.func @test_fold_mod_arith_add_with_reduction() -> !mod_poly_ty {
  // CHECK-NOT: polynomial.add
  // CHECK: %[[CST:.*]] = polynomial.constant int<2 + 2x> : [[TY_MOD_ADD:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_ADD]]
  %0 = polynomial.constant int<10 + 16x> : !mod_poly_ty
  %1 = polynomial.constant int<9 + 3x> : !mod_poly_ty
  %2 = polynomial.add %0, %1 : !mod_poly_ty
  return %2 : !mod_poly_ty
}

// CHECK: @test_fold_mod_arith_sub_with_reduction
func.func @test_fold_mod_arith_sub_with_reduction() -> !mod_poly_ty {
  // CHECK-NOT: polynomial.sub
  // CHECK: %[[CST:.*]] = polynomial.constant int<13 + 14x> : [[TY_MOD_SUB:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_SUB]]
  %0 = polynomial.constant int<1 + x> : !mod_poly_ty
  %1 = polynomial.constant int<5 + 4x> : !mod_poly_ty
  %2 = polynomial.sub %0, %1 : !mod_poly_ty
  return %2 : !mod_poly_ty
}

// CHECK: @test_fold_mod_arith_mul_eval
func.func @test_fold_mod_arith_mul_eval() -> !mod_int_poly_ev_ty {
  // CHECK-NOT: polynomial.mul
  // CHECK: %[[CST:.*]] = polynomial.constant int<14 + 16x> : [[TY_MOD_MUL_EVAL:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_MUL_EVAL]]
  %0 = polynomial.constant int<2 + 3x + 5x**2> : !mod_int_poly_ev_ty
  %1 = polynomial.constant int<7 + 11x + 13x**3> : !mod_int_poly_ev_ty
  %2 = polynomial.mul %0, %1 : !mod_int_poly_ev_ty
  return %2 : !mod_int_poly_ev_ty
}

// CHECK: @test_fold_mod_arith_mul_coeff
func.func @test_fold_mod_arith_mul_coeff() -> !mod_int_poly_ty {
  // CHECK-NOT: polynomial.mul
  // CHECK: %[[CST:.*]] = polynomial.constant int<3 + 9x + 4x**2 + 6x**3> : [[TY_MOD_MUL_COEFF:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_MUL_COEFF]]
  %0 = polynomial.constant int<1 + 2x**3> : !mod_int_poly_ty
  %1 = polynomial.constant int<3 + 4x**2> : !mod_int_poly_ty
  %2 = polynomial.mul %0, %1 : !mod_int_poly_ty
  return %2 : !mod_int_poly_ty
}

// CHECK: @test_fold_mod_arith_monomial
func.func @test_fold_mod_arith_monomial() -> !mod_poly_ty {
  // CHECK-NOT: polynomial.monomial
  // CHECK: %[[CST:.*]] = polynomial.constant int<4x> : [[TY_MOD_MONOMIAL:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_MONOMIAL]]
  %degree = arith.constant 1 : index
  %coeff = mod_arith.constant 4 : !rns_basis_0
  %0 = polynomial.monomial %coeff, %degree : (!rns_basis_0, index) -> !mod_poly_ty
  return %0 : !mod_poly_ty
}

// CHECK: @test_fold_mod_arith_monic_monomial_mul
func.func @test_fold_mod_arith_monic_monomial_mul() -> !mod_poly_ty {
  // CHECK-NOT: polynomial.monic_monomial_mul
  // CHECK: %[[CST:.*]] = polynomial.constant int<14> : [[TY_MOD_MONIC_MONOMIAL_MUL:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_MONIC_MONOMIAL_MUL]]
  %degree = arith.constant 1 : index
  %0 = polynomial.constant int<20x> : !mod_poly_ty
  %1 = polynomial.monic_monomial_mul %0, %degree : (!mod_poly_ty, index) -> !mod_poly_ty
  return %1 : !mod_poly_ty
}

// CHECK: @test_fold_mod_arith_mul_scalar
func.func @test_fold_mod_arith_mul_scalar() -> !mod_poly_ty {
  // CHECK-NOT: polynomial.mul_scalar
  // CHECK: %[[CST:.*]] = polynomial.constant int<9 + 8x> : [[TY_MOD_MUL_SCALAR:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_MUL_SCALAR]]
  %scalar = mod_arith.constant 3 : !rns_basis_0
  %0 = polynomial.constant int<20 + 14x> : !mod_poly_ty
  %1 = polynomial.mul_scalar %0, %scalar : !mod_poly_ty, !rns_basis_0
  return %1 : !mod_poly_ty
}

// CHECK: @test_fold_mod_arith_ntt
func.func @test_fold_mod_arith_ntt() -> !mod_poly_ev_ty {
  // CHECK-NOT: polynomial.ntt
  // CHECK: %[[CST:.*]] = polynomial.constant int<9 + 10x> : [[TY_MOD_NTT:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_NTT]]
  %0 = polynomial.constant int<1 + 2x> : !mod_poly_ty
  %1 = polynomial.ntt %0 {root = #mod_root} : !mod_poly_ty
  return %1 : !mod_poly_ev_ty
}

// CHECK: @test_fold_mod_arith_intt
func.func @test_fold_mod_arith_intt() -> !mod_poly_ty {
  // CHECK-NOT: polynomial.intt
  // CHECK: %[[CST:.*]] = polynomial.constant int<10 + 2x> : [[TY_MOD_INTT:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_MOD_INTT]]
  %0 = polynomial.constant int<1 + 2x> : !mod_poly_ev_ty
  %1 = polynomial.intt %0 {root = #mod_root} : !mod_poly_ev_ty
  return %1 : !mod_poly_ty
}





// CHECK: @test_fold_int_add_eval
func.func @test_fold_int_add_eval() -> !int_poly_ev_ty {
  // CHECK-NOT: polynomial.add
  // CHECK: %[[CST:.*]] = polynomial.constant int<4 + 2x + 4x**2> : [[TY_INT_ADD:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_INT_ADD]]
  %0 = polynomial.constant int<1 + 2x> : !int_poly_ev_ty
  %1 = polynomial.constant int<3 + 4x**2> : !int_poly_ev_ty
  %2 = polynomial.add %0, %1 : !int_poly_ev_ty
  return %2 : !int_poly_ev_ty
}

// CHECK: @test_fold_int_sub_eval
func.func @test_fold_int_sub_eval() -> !int_poly_ev_ty {
  // CHECK-NOT: polynomial.sub
  // CHECK: %[[CST:.*]] = polynomial.constant int<3 + 4x> : [[TY_INT_SUB:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_INT_SUB]]
  %0 = polynomial.constant int<5 + 7x> : !int_poly_ev_ty
  %1 = polynomial.constant int<2 + 3x> : !int_poly_ev_ty
  %2 = polynomial.sub %0, %1 : !int_poly_ev_ty
  return %2 : !int_poly_ev_ty
}

// CHECK: @test_fold_int_mul_coeff
func.func @test_fold_int_mul_coeff() -> !int_poly_ty {
  // CHECK-NOT: polynomial.mul
  // CHECK: %[[CST:.*]] = polynomial.constant int<3 + 4x**2 + 6x**3 + 8x**5> : [[TY_INT_MUL_COEFF:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_INT_MUL_COEFF]]
  %0 = polynomial.constant int<1 + 2x**3> : !int_poly_ty
  %1 = polynomial.constant int<3 + 4x**2> : !int_poly_ty
  %2 = polynomial.mul %0, %1 : !int_poly_ty
  return %2 : !int_poly_ty
}

// CHECK: @test_fold_int_monomial
func.func @test_fold_int_monomial() -> !int_poly_ty {
  // CHECK-NOT: polynomial.monomial
  // CHECK: %[[CST:.*]] = polynomial.constant int<4x> : [[TY_INT_MONOMIAL:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_INT_MONOMIAL]]
  %degree = arith.constant 1 : index
  %coeff = arith.constant 4 : i32
  %0 = polynomial.monomial %coeff, %degree : (i32, index) -> !int_poly_ty
  return %0 : !int_poly_ty
}

// CHECK: @test_fold_int_monic_monomial_mul
func.func @test_fold_int_monic_monomial_mul() -> !int_poly_ty {
  // CHECK-NOT: polynomial.monic_monomial_mul
  // CHECK: %[[CST:.*]] = polynomial.constant int<-2x + x**2> : [[TY_INT_MONIC_MONOMIAL_MUL:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_INT_MONIC_MONOMIAL_MUL]]
  %degree = arith.constant 2 : index
  %0 = polynomial.constant int<1 + 2x**3> : !int_poly_ty
  %1 = polynomial.monic_monomial_mul %0, %degree : (!int_poly_ty, index) -> !int_poly_ty
  return %1 : !int_poly_ty
}

// CHECK: @test_fold_int_mul_scalar
func.func @test_fold_int_mul_scalar() -> !int_poly_ty {
  // CHECK-NOT: polynomial.mul_scalar
  // CHECK: %[[CST:.*]] = polynomial.constant int<3 + 6x**3> : [[TY_INT_MUL_SCALAR:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_INT_MUL_SCALAR]]
  %scalar = arith.constant 3 : i32
  %0 = polynomial.constant int<1 + 2x**3> : !int_poly_ty
  %1 = polynomial.mul_scalar %0, %scalar : !int_poly_ty, i32
  return %1 : !int_poly_ty
}

// CHECK: @test_fold_float_mul_scalar
func.func @test_fold_float_mul_scalar() -> !float_poly_ty {
  // CHECK-NOT: polynomial.mul_scalar
  // CHECK: %[[CST:.*]] = polynomial.constant float<3 + 6x**2> : [[TY_FLOAT_MUL_SCALAR:![a-zA-Z0-9_]+]]
  // CHECK: return %[[CST]] : [[TY_FLOAT_MUL_SCALAR]]
  %scalar = arith.constant 3.0 : f32
  %0 = polynomial.constant float<1.0 + 2.0 x**2> : !float_poly_ty
  %1 = polynomial.mul_scalar %0, %scalar : !float_poly_ty, f32
  return %1 : !float_poly_ty
}
