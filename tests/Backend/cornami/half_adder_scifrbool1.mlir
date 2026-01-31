#unspecified_bit_field_encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
module {
  func.func @half_adder(%arg0: !scifrbool.bootstrap_key_standard, %arg1: !scifrbool.key_switch_key, %arg2: !scifrbool.server_parameters, %ct: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, %ct_0: !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> tensor<2x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>> {
    %ct_1 = scifrbool.section(%ct, %ct_0) {
      %ct_3 = scifrbool.xor %ct, %ct_0 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %ct_2 = scifrbool.section(%ct, %ct_0) {
      %ct_3 = scifrbool.and %ct, %ct_0 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    } : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %from_elements = tensor.from_elements %ct_1, %ct_2 : tensor<2x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    return %from_elements : tensor<2x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
  }
}
