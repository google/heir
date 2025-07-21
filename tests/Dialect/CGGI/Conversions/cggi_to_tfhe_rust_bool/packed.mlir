// RUN: heir-opt --cggi-to-tfhe-rust-bool %s | FileCheck %s

#key = #lwe.key<slot_index = 0>
#preserve_overflow = #lwe.preserve_overflow<>
#app_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i1, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!pt_ty = !lwe.lwe_plaintext<application_data = #app_data, plaintext_space = #pspace>
!ct_ty = !lwe.lwe_ciphertext<application_data = #app_data, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: packed
// CHECK-NOT: cggi
// CHECK-NOT: lwe
func.func @packed(%arg0: tensor<8x!ct_ty>, %arg1: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
  %ha_1 = cggi.xor %arg0, %arg1 : tensor<8x!ct_ty>
  %ha_2 = cggi.and %ha_1, %arg1 : tensor<8x!ct_ty>
  %ha_3 = cggi.nand %ha_2, %arg1 : tensor<8x!ct_ty>
  %ha_4 = cggi.xnor %ha_3, %arg1 : tensor<8x!ct_ty>
  %ha_5 = cggi.or %ha_4, %arg1 : tensor<8x!ct_ty>
  %ha_6 = cggi.not %ha_5 : tensor<8x!ct_ty>
  return %ha_6 : tensor<8x!ct_ty>
}
