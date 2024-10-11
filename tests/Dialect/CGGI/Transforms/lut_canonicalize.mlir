// RUN: heir-opt --canonicalize %s | FileCheck %s
#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 3>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>

func.func @require_post_pass_toposort_lut3(%arg0: tensor<8x!ct_ty>) -> !ct_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %1 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %2 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>

  // CHECK: cggi.lut_lincomb %extracted, %extracted_0, %extracted_1 {coefficients = array<i32: 1, 2, 4>, lookup_table = 8 : ui8}
  %r1 = cggi.lut3 %0, %1, %2 {lookup_table = 8 : ui8} : !ct_ty

  return %r1 : !ct_ty
}

func.func @require_post_pass_toposort_lut2(%arg0: tensor<8x!ct_ty>) -> !ct_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %1 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %2 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>

  // CHECK: cggi.lut_lincomb %extracted, %extracted_0 {coefficients = array<i32: 1, 2>, lookup_table = 8 : ui8}
  %r1 = cggi.lut2 %0, %1 {lookup_table = 8 : ui8} : !ct_ty

  return %r1 : !ct_ty
}
