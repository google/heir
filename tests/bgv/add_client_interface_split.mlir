// RUN: heir-opt '--bgv-add-client-interface=use-public-key=true one-value-per-helper-fn=true' %s | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
#params = #lwe.rlwe_params<ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**8>>>
!in_ty = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = tensor<8xi16>>
!out_ty = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = i16>
!mul_ty = !lwe.rlwe_ciphertext<encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, rlwe_params = <dimension = 3, ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**8>>>, underlying_type = tensor<8xi16>>

func.func @dot_product(%arg0: !in_ty, %arg1: !in_ty) -> (!out_ty, !out_ty) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = bgv.mul %arg0, %arg1 : (!in_ty, !in_ty) -> !mul_ty
  %1 = bgv.relinearize %0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !mul_ty -> !in_ty
  %2 = bgv.rotate %1, %c4 : !in_ty, index
  %3 = bgv.add %1, %2 : !in_ty
  %4 = bgv.rotate %3, %c2 : !in_ty, index
  %5 = bgv.add %3, %4 : !in_ty
  %6 = bgv.rotate %5, %c1 : !in_ty, index
  %7 = bgv.add %5, %6 : !in_ty
  %8 = bgv.extract %7, %c7 : (!in_ty, index) -> !out_ty
  return %8, %8 : !out_ty, !out_ty
}

// CHECK: func.func @dot_product__encrypt__arg0
// CHECK: func.func @dot_product__encrypt__arg1
// CHECK: func.func @dot_product__decrypt__result0
// CHECK: func.func @dot_product__decrypt__result1
