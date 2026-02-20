#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>
!bsks = !scifrbool.bootstrap_key_standard
!ksk = !scifrbool.key_switch_key
!params = !scifrbool.server_parameters

func.func @half_adder(%arg0: !ct_ty, %arg1: !ct_ty, %bsks: !bsks, %ksk: !ksk, %params: !params) -> tensor<2x!ct_ty> {
  %0 = scifrbool.xor %arg0, %arg1 : !ct_ty
  %1 = scifrbool.and %arg0, %arg1 : !ct_ty
  %from_elements = tensor.from_elements %0, %1 : tensor<2x!ct_ty>

  return %from_elements : tensor<2x!ct_ty>
}
