// RUN: heir-opt --cggi-to-tfhe-rust-bool -cse -remove-dead-values %s | FileCheck %s

#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>

// CHECK-LABEL: add_bool
// CHECK-NOT: cggi
// CHECK-NOT: lwe
func.func @add_bool(%arg0: tensor<8x!ct_ty>, %arg1: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
  %true = arith.constant true
  %false = arith.constant false
  %c7 = arith.constant 7 : index
  %c6 = arith.constant 6 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %extracted_00 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %extracted_01 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %extracted_02 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>
  %extracted_03 = tensor.extract %arg0[%c3] : tensor<8x!ct_ty>
  %extracted_04 = tensor.extract %arg0[%c4] : tensor<8x!ct_ty>
  %extracted_05 = tensor.extract %arg0[%c5] : tensor<8x!ct_ty>
  %extracted_06 = tensor.extract %arg0[%c6] : tensor<8x!ct_ty>
  %extracted_07 = tensor.extract %arg0[%c7] : tensor<8x!ct_ty>
  %extracted_10 = tensor.extract %arg1[%c0] : tensor<8x!ct_ty>
  %extracted_11 = tensor.extract %arg1[%c1] : tensor<8x!ct_ty>
  %extracted_12 = tensor.extract %arg1[%c2] : tensor<8x!ct_ty>
  %extracted_13 = tensor.extract %arg1[%c3] : tensor<8x!ct_ty>
  %extracted_14 = tensor.extract %arg1[%c4] : tensor<8x!ct_ty>
  %extracted_15 = tensor.extract %arg1[%c5] : tensor<8x!ct_ty>
  %extracted_16 = tensor.extract %arg1[%c6] : tensor<8x!ct_ty>
  %extracted_17 = tensor.extract %arg1[%c7] : tensor<8x!ct_ty>
  %0 = cggi.xor %extracted_00, %extracted_10 : !ct_ty
  %1 = cggi.and %extracted_02, %extracted_12 : !ct_ty
  %2 = cggi.xor %extracted_01, %extracted_11 : !ct_ty
  %3 = cggi.and %extracted_03, %extracted_13 : !ct_ty
  %4 = cggi.xor %extracted_05, %extracted_15 : !ct_ty
  %5 = cggi.and %extracted_04, %extracted_14 : !ct_ty
  %6 = cggi.xor %extracted_06, %extracted_16 : !ct_ty
  %7 = cggi.and %extracted_07, %extracted_17 : !ct_ty
  %from_elements = tensor.from_elements %0, %2, %4, %6, %1, %3, %5, %7 : tensor<8x!ct_ty>
  return %from_elements : tensor<8x!ct_ty>
}
