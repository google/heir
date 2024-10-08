// RUN: heir-translate %s --emit-tfhe-rust-bool | FileCheck %s

!bsks = !tfhe_rust_bool.server_key
!eb = !tfhe_rust_bool.eb

// CHECK-LABEL: pub fn test_and(
// CHECK-NEXT:   [[bsks:v[0-9]+]]: &ServerKey,
// CHECK-NEXT:   [[input1:v[0-9]+]]: &Ciphertext,
// CHECK-NEXT:   [[input2:v[0-9]+]]: &Ciphertext,
// CHECK-NEXT: ) -> Ciphertext {
// CHECK-NEXT:   let [[v0:.*]] = [[bsks]].and([[input1]], [[input2]]);
// CHECK-NEXT:   [[v0]]
// CHECK-NEXT: }
func.func @test_and(%bsks : !bsks, %input1 : !eb, %input2 : !eb) -> !eb {
  %out = tfhe_rust_bool.and %bsks, %input1, %input2 : (!bsks, !eb, !eb) -> !eb
  return %out : !eb
}
