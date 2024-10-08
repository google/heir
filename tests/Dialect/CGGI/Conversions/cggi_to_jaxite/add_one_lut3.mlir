// RUN: heir-opt --cggi-to-jaxite %s | FileCheck %s

#unspecified_encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 3>
!ct_ty = !lwe.lwe_ciphertext<encoding = #unspecified_encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #unspecified_encoding>


// CHECK-LABEL: test_add_one_lut3
// CHECK-COUNT-2: jaxite.constant
// CHECK-NOT: lwe.trivial_encrypt
// CHECK-COUNT-11: jaxite.lut3
// CHECK-NOT: cggi.lut3
func.func @test_add_one_lut3(%arg0: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
  %c7 = arith.constant 7 : index
  %c6 = arith.constant 6 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %x_00 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %x_01 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %x_02 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>
  %x_03 = tensor.extract %arg0[%c3] : tensor<8x!ct_ty>
  %x_04 = tensor.extract %arg0[%c4] : tensor<8x!ct_ty>
  %x_05 = tensor.extract %arg0[%c5] : tensor<8x!ct_ty>
  %x_06 = tensor.extract %arg0[%c6] : tensor<8x!ct_ty>
  %x_07 = tensor.extract %arg0[%c7] : tensor<8x!ct_ty>

  %bool_0 = arith.constant 0 : i1
  %bool_1 = arith.constant 1 : i1


  %encoded_1 = lwe.encode %bool_1 {encoding = #unspecified_encoding} : i1 to !pt_ty
  %constant_T = lwe.trivial_encrypt %encoded_1 : !pt_ty to !ct_ty
  %encoded_0 = lwe.encode %bool_0 {encoding = #unspecified_encoding} : i1 to !pt_ty
  %constant_F = lwe.trivial_encrypt %encoded_0 : !pt_ty to !ct_ty

  %t_0 = cggi.lut3 %x_00, %x_01, %x_02 {lookup_table = 128 : ui8} : !ct_ty
  %t_1 = cggi.lut3 %t_0, %x_03, %x_04 {lookup_table = 128 : ui8} : !ct_ty
  %t_2 = cggi.lut3 %t_1, %x_05, %x_06 {lookup_table = 128 : ui8} : !ct_ty


  %res_07 = cggi.lut3 %t_2, %x_07, %constant_F {lookup_table = 6 : ui8} : !ct_ty
  %res_06 = cggi.lut3 %t_1, %x_05, %x_06 {lookup_table = 120 : ui8} : !ct_ty
  %res_05 = cggi.lut3 %t_1, %x_05, %constant_F {lookup_table = 6 : ui8} : !ct_ty
  %res_04 = cggi.lut3 %t_0, %x_03, %x_04 {lookup_table = 120 : ui8} : !ct_ty
  %res_03 = cggi.lut3 %t_0, %x_03, %constant_F {lookup_table = 6 : ui8} : !ct_ty
  %res_02 = cggi.lut3 %x_00, %x_01, %x_02 {lookup_table = 120 : ui8} : !ct_ty
  %res_01 = cggi.lut3 %x_00, %x_01, %constant_F {lookup_table = 6 : ui8} : !ct_ty
  %res_00 = cggi.lut3 %x_00, %constant_F, %constant_F {lookup_table =1 : ui8} : !ct_ty

  %from_elements = tensor.from_elements %res_00, %res_01, %res_02, %res_03, %res_04, %res_05, %res_06, %res_07 : tensor<8x!ct_ty>
  return %from_elements : tensor<8x!ct_ty>
}
