// RUN: heir-translate --emit-jaxite %s | FileCheck %s

!bsks = !jaxite.server_key_set
!params = !jaxite.params

#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!eb = !lwe.lwe_ciphertext<plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: def test_add_one_lut3(
// CHECK-NEXT:   [[v0:.*]]: list[types.LweCiphertext],
// CHECK-NEXT:   [[v1:.*]]: jaxite_bool.ServerKeySet,
// CHECK-NEXT:   [[v2:.*]]: jaxite_bool.Parameters,
// CHECK-NEXT: ) -> list[types.LweCiphertext]:
// CHECK-COUNT-2: jaxite_bool.constant
// CHECK-NOT: jaxite.constant
// CHECK-COUNT-11: jaxite_bool.lut3
// CHECK-NOT: jaxite.lut3
func.func @test_add_one_lut3(%arg0: tensor<8x!eb>, %bsks : !bsks, %params : !params) -> tensor<8x!eb> {
  %c7 = arith.constant 7 : index
  %c6 = arith.constant 6 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %x_00 = tensor.extract %arg0[%c0] : tensor<8x!eb>
  %x_01 = tensor.extract %arg0[%c1] : tensor<8x!eb>
  %x_02 = tensor.extract %arg0[%c2] : tensor<8x!eb>
  %x_03 = tensor.extract %arg0[%c3] : tensor<8x!eb>
  %x_04 = tensor.extract %arg0[%c4] : tensor<8x!eb>
  %x_05 = tensor.extract %arg0[%c5] : tensor<8x!eb>
  %x_06 = tensor.extract %arg0[%c6] : tensor<8x!eb>
  %x_07 = tensor.extract %arg0[%c7] : tensor<8x!eb>

  %bool_0 = arith.constant 0 : i1
  %bool_1 = arith.constant 1 : i1
  %tt_1 = arith.constant 1 : i8
  %tt_6 = arith.constant 6 : i8
  %tt_120 = arith.constant 120 : i8
  %tt_128 = arith.constant 128 : i8

  %constant_T = jaxite.constant %bool_1, %params : (i1, !params) -> !eb
  %constant_F = jaxite.constant %bool_0, %params : (i1, !params) -> !eb

  %t_0 = jaxite.lut3 %x_00, %x_01, %x_02, %tt_128, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %t_1 = jaxite.lut3 %t_0, %x_03, %x_04, %tt_128, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %t_2 = jaxite.lut3 %t_1, %x_05, %x_06, %tt_128, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb

  %res_07 = jaxite.lut3 %t_2, %x_07, %constant_F, %tt_6, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %res_06 = jaxite.lut3 %t_1, %x_05, %x_06, %tt_120, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %res_05 = jaxite.lut3 %t_1, %x_05, %constant_F, %tt_6, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %res_04 = jaxite.lut3 %t_0, %x_03, %x_04, %tt_120, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %res_03 = jaxite.lut3 %t_0, %x_03, %constant_F, %tt_6, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %res_02 = jaxite.lut3 %x_00, %x_01, %x_02, %tt_120, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %res_01 = jaxite.lut3 %x_00, %x_01, %constant_F, %tt_6, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb
  %res_00 = jaxite.lut3 %x_00, %constant_F, %constant_F, %tt_1, %bsks, %params : (!eb, !eb, !eb, i8, !bsks, !params) -> !eb

  %from_elements = tensor.from_elements %res_00, %res_01, %res_02, %res_03, %res_04, %res_05, %res_06, %res_07 : tensor<8x!eb>
  return %from_elements : tensor<8x!eb>
}
