#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>

func.func @half_adder(%arg0: !ct_ty, %arg1: !ct_ty) -> tensor<2x!ct_ty> {
  %0 = cggi.xor %arg0, %arg1 : !ct_ty
  %1 = cggi.and %arg0, %arg1 : !ct_ty
  %from_elements = tensor.from_elements %0, %1 : tensor<2x!ct_ty>

  return %from_elements : tensor<2x!ct_ty>
}
