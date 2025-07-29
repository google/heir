// RUN: heir-translate --emit-jaxite %s | FileCheck %s

!bsks = !jaxite.server_key_set
!params = !jaxite.params

#key = #lwe.key<slot_index = 0>
#preserve_overflow = #lwe.preserve_overflow<>
#app_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!eb = !lwe.lwe_ciphertext<application_data = #app_data, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: def test_return_multiple_values(
// CHECK-NEXT:   [[input:ct]]: types.LweCiphertext,
// CHECK-NEXT:   [[v1:.*]]: jaxite_bool.ServerKeySet,
// CHECK-NEXT:   [[v2:.*]]: jaxite_bool.Parameters,
// CHECK-NEXT: ) -> (types.LweCiphertext, types.LweCiphertext):
// CHECK:   return ([[input]], [[input]])
func.func @test_return_multiple_values(%input: !eb, %bsks : !bsks, %params : !params) -> (!eb, !eb) {
  return %input, %input : !eb, !eb
}


// CHECK: def test_memref_load(
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

// CHECK: def test_memref_alloc_store(
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

// CHECK: def test_packed_lut3(
// CHECK-NEXT:   [[sk:.*]]: jaxite_bool.ServerKeySet,
// CHECK-NEXT:   [[params:.*]]: jaxite_bool.Parameters
// CHECK-NEXT: ):
// CHECK-NEXT:  temp_nodes: Dict[int, Any] = {}
// CHECK-NEXT:  [[e1:temp_nodes[[0-9]+]]] = jaxite_bool.constant(True,[[params]])
// CHECK-NEXT:  [[e2:temp_nodes[[0-9]+]]] = jaxite_bool.constant(True,[[params]])
// CHECK-NEXT:  [[e3:temp_nodes[[0-9]+]]] = jaxite_bool.constant(True,[[params]])
// CHECK-NEXT:  inputs = [([[e1]], [[e2]], [[e3]], 8),]
// CHECK-NEXT:  [[e4:temp_nodes[[0-9]+]]] = jaxite_bool.pmap_lut3(inputs,[[sk]],[[params]])
func.func @test_packed_lut3(%bsks : !bsks, %params : !params) {
    %0 = arith.constant 1 : i1
    %1 = arith.constant 1 : i1
    %2 = arith.constant 1 : i1
    %truth_table = arith.constant 8 : i8

    %e1 = jaxite.constant %0, %params : (i1, !params) -> !eb
    %e2 = jaxite.constant %1, %params : (i1, !params) -> !eb
    %e3 = jaxite.constant %2, %params : (i1, !params) -> !eb

    %l1 = jaxite.lut3_args %e1, %e2, %e3, %truth_table :(!eb, !eb, !eb, i8) -> !jaxite.pmap_lut3_tuple
    %lut3_args_tensor = tensor.from_elements %l1 : tensor<1x!jaxite.pmap_lut3_tuple>

    %out = jaxite.pmap_lut3 %lut3_args_tensor, %bsks, %params : (tensor<1x!jaxite.pmap_lut3_tuple>, !bsks, !params) -> !eb
    return
  }
