// RUN: heir-opt --cggi-boolean-line-vectorize %s | FileCheck %s

#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>

// CHECK-LABEL: add_one
// CHECK-COUNT-1: cggi.packed_gates %from_elements, %from_elements_15 {gates = #cggi.cggi_gate<"and", "xor">} : (tensor<2x!lwe.lwe_ciphertext<encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>>>, tensor<2x!lwe.lwe_ciphertext<encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>>>) -> tensor<2x!lwe.lwe_ciphertext<encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>>>
func.func @add_one(%arg0: tensor<8x!ct_ty>, %arg1: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
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
  %ha_s = cggi.xor %extracted_00, %extracted_10 : !ct_ty
  %ha_c = cggi.and %extracted_00, %extracted_10 : !ct_ty
  %fa0_1 = cggi.xor %extracted_01, %extracted_11 : !ct_ty
  %fa0_2 = cggi.and %extracted_01, %extracted_11 : !ct_ty
  %fa0_3 = cggi.and %fa0_1, %ha_c : !ct_ty
  %fa0_s = cggi.xor %fa0_1, %ha_c : !ct_ty
  %fa0_c = cggi.xor %fa0_2, %fa0_3 : !ct_ty
  %fa1_1 = cggi.xor %extracted_02, %extracted_12 : !ct_ty
  %fa1_2 = cggi.and %extracted_02, %extracted_12 : !ct_ty
  %fa1_3 = cggi.and %fa1_1, %fa0_c : !ct_ty
  %fa1_s = cggi.xor %fa1_1, %fa0_c : !ct_ty
  %fa1_c = cggi.xor %fa1_2, %fa1_3 : !ct_ty
  %fa2_1 = cggi.xor %extracted_03, %extracted_13 : !ct_ty
  %fa2_2 = cggi.and %extracted_03, %extracted_13 : !ct_ty
  %fa2_3 = cggi.and %fa2_1, %fa1_c : !ct_ty
  %fa2_s = cggi.xor %fa2_1, %fa1_c : !ct_ty
  %fa2_c = cggi.xor %fa2_2, %fa2_3 : !ct_ty
  %fa3_1 = cggi.xor %extracted_04, %extracted_14 : !ct_ty
  %fa3_2 = cggi.and %extracted_04, %extracted_14 : !ct_ty
  %fa3_3 = cggi.and %fa3_1, %fa2_c : !ct_ty
  %fa3_s = cggi.xor %fa3_1, %fa2_c : !ct_ty
  %fa3_c = cggi.xor %fa3_2, %fa3_3 : !ct_ty
  %fa4_1 = cggi.xor %extracted_05, %extracted_15 : !ct_ty
  %fa4_2 = cggi.and %extracted_05, %extracted_15 : !ct_ty
  %fa4_3 = cggi.and %fa4_1, %fa3_c : !ct_ty
  %fa4_s = cggi.xor %fa4_1, %fa3_c : !ct_ty
  %fa4_c = cggi.xor %fa4_2, %fa4_3 : !ct_ty
  %fa5_1 = cggi.xor %extracted_06, %extracted_16 : !ct_ty
  %fa5_2 = cggi.and %extracted_06, %extracted_16 : !ct_ty
  %fa5_3 = cggi.and %fa5_1, %fa4_c : !ct_ty
  %fa5_s = cggi.xor %fa5_1, %fa4_c : !ct_ty
  %fa5_c = cggi.xor %fa5_2, %fa5_3 : !ct_ty
  %fa6_1 = cggi.xor %extracted_07, %extracted_17 : !ct_ty
  %fa6_2 = cggi.and %extracted_07, %extracted_17 : !ct_ty
  %fa6_3 = cggi.and %fa6_1, %fa5_c : !ct_ty
  %fa6_s = cggi.xor %fa6_1, %fa5_c : !ct_ty
  %fa6_c = cggi.xor %fa6_2, %fa6_3 : !ct_ty
  %from_elements = tensor.from_elements %fa6_s, %fa5_s, %fa4_s, %fa3_s, %fa2_s, %fa1_s, %fa0_s, %ha_s : tensor<8x!ct_ty>
  return %from_elements : tensor<8x!ct_ty>
}
