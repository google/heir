// RUN: heir-opt --cggi-to-tfhe-rust-bool %s | FileCheck %s

#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>

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
