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


// CHECK-LABEL: def test_memref_load(
// CHECK-NEXT:   [[input:v[0-9]+]]: list[types.LweCiphertext],
// CHECK-NEXT:   [[v1:.*]]: jaxite_bool.ServerKeySet,
// CHECK-NEXT:   [[v2:.*]]: jaxite_bool.Parameters,
// CHECK-NEXT: ) -> types.LweCiphertext:
// CHECK-NEXT:  temp_nodes: Dict[int, Any] = {}
// CHECK-NEXT:  [[output:temp_nodes[[0-9]+]]] = [[input]][0]
// CHECK-NEXT:  return [[output]]
func.func @test_memref_load(%input: memref<8x!eb>, %bsks : !bsks, %params : !params) -> !eb {
  %c0 = arith.constant 0 : index
  %0 = memref.load %input[%c0] : memref<8x!eb>
  return %0 : !eb
}

// CHECK-LABEL: def test_memref_alloc_store(
// CHECK-NEXT:   [[input:v[0-9]+]]: list[types.LweCiphertext],
// CHECK-NEXT:   [[v1:.*]]: jaxite_bool.ServerKeySet,
// CHECK-NEXT:   [[v2:.*]]: jaxite_bool.Parameters,
// CHECK-NEXT: ) -> list[types.LweCiphertext]:
// CHECK-NEXT:  temp_nodes: Dict[int, Any] = {}
// CHECK-NEXT:  [[input0:temp_nodes[[0-9]+]]] = [[input]][0]
// CHECK-NEXT:  [[input1:temp_nodes[[0-9]+]]] = [[input]][1]
// CHECK-NEXT:  [[alloc:temp_nodes[[0-9]+]]] = np.full((2), None)
// CHECK-NEXT:  [[alloc]][0] = [[input0]]
// CHECK-NEXT:  [[alloc]][1] = [[input1]]
// CHECK-NEXT:  return [[alloc]]
func.func @test_memref_alloc_store(%input: memref<8x!eb>, %bsks : !bsks, %params : !params) -> memref<2x!eb> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.load %input[%c0] : memref<8x!eb>
  %1 = memref.load %input[%c1] : memref<8x!eb>
  %alloc = memref.alloc() : memref<2x!eb>
  memref.store %0, %alloc[%c0] : memref<2x!eb>
  memref.store %1, %alloc[%c1] : memref<2x!eb>
  return %alloc : memref<2x!eb>
}
