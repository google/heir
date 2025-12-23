// RUN: heir-translate %s --emit-tfhe-rust-bool-packed | FileCheck %s

!bsks = !tfhe_rust_bool.server_key
!eb = !tfhe_rust_bool.eb

module{
// CHECK: pub fn test_and(
// CHECK-NEXT:   [[bsks:v[0-9]+]]: &ServerKey,
// CHECK-NEXT:   [[input1:v[0-9]+]]: &[Ciphertext; 4],
// CHECK-NEXT:   [[input2:v[0-9]+]]: &[Ciphertext; 4],
// CHECK-NEXT: ) -> [Ciphertext; 4] {
// CHECK-NEXT:  let [[v0:.*]] = [[bsks]].packed_gates(
// CHECK-NEXT:   &vec![Gate::AND, Gate::AND, Gate::AND, Gate::AND, ],
// CHECK-NEXT:  &[[input1]].each_ref(), &[[input2]].each_ref());
// CHECK-NEXT:   [[v0]]
// CHECK-NEXT: }
func.func @test_and(%bsks : !bsks, %input1 : tensor<4x!eb>, %input2 : tensor<4x!eb>) -> tensor<4x!eb> {
  %out = tfhe_rust_bool.and %bsks, %input1, %input2 : (!bsks, tensor<4x!eb>, tensor<4x!eb>) -> tensor<4x!eb>
  return %out : tensor<4x!eb>
}

// CHECK: pub fn test_not(
// CHECK-NEXT:   [[bsks:v[0-9]+]]: &ServerKey,
// CHECK-NEXT:   [[input1:v[0-9]+]]: &[Ciphertext; 4],
// CHECK-NEXT: ) -> [Ciphertext; 4] {
// CHECK-NEXT:  let [[v0:.*]] = [[bsks]].packed_not(
// CHECK-NEXT:  &[[input1]].each_ref());
// CHECK-NEXT:   [[v0]]
// CHECK-NEXT: }
func.func @test_not(%bsks : !bsks, %input1 : tensor<4x!eb>) -> tensor<4x!eb>{
  %out = tfhe_rust_bool.not %bsks, %input1 : (!bsks, tensor<4x!eb>) -> tensor<4x!eb>
  return %out : tensor<4x!eb>
}

}
