// RUN: heir-opt --cggi-decompose-operations=expand-lincomb=false %s | FileCheck %s

#key = #lwe.key<slot_index = 0>
#preserve_overflow = #lwe.preserve_overflow<>
#app_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!pt_ty = !lwe.lwe_plaintext<application_data = #app_data, plaintext_space = #pspace>
!ct_ty = !lwe.lwe_ciphertext<application_data = #app_data, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

func.func @require_post_pass_toposort_lut3(%arg0: tensor<8x!ct_ty>) -> !ct_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %1 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %2 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>

  // CHECK: cggi.lut_lincomb %extracted, %extracted_0, %extracted_1 {coefficients = array<i32: 4, 2, 1>, lookup_table = 8 : ui8}
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

  // CHECK: cggi.lut_lincomb %extracted, %extracted_0 {coefficients = array<i32: 2, 1>, lookup_table = 8 : ui8}
  %r1 = cggi.lut2 %0, %1 {lookup_table = 8 : ui8} : !ct_ty

  return %r1 : !ct_ty
}
