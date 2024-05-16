// RUN: heir-translate --emit-jaxite %s | FileCheck %s

!bsks = !jaxite.server_key_set
!params = !jaxite.params
#unspecified_encoding = #lwe.unspecified_bit_field_encoding<
  cleartext_bitwidth=3>
#params = #lwe.lwe_params<cmod=7917, dimension=10>
!eb = !lwe.lwe_ciphertext<encoding = #unspecified_encoding, lwe_params = #params>


// CHECK-LABEL: def test_return_multiple_values(
// CHECK-NEXT:   [[input:v[0-9]+]]: types.LweCiphertext,
// CHECK-NEXT:   [[v1:.*]]: jaxite_bool.ServerKeySet,
// CHECK-NEXT:   [[v2:.*]]: jaxite_bool.Parameters,
// CHECK-NEXT: ) -> (types.LweCiphertext, types.LweCiphertext):
// CHECK:   return ([[input]], [[input]])
func.func @test_return_multiple_values(%input: !eb, %bsks : !bsks, %params : !params) -> (!eb, !eb) {
  return %input, %input : !eb, !eb
}
