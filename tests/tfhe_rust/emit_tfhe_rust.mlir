// RUN: heir-translate %s --emit-tfhe-rust | FileCheck %s

!sks = !tfhe_rust.server_key
!lut = !tfhe_rust.lookup_table
!eui3 = !tfhe_rust.eui3

// CHECK-LABEL: pub fn test_apply_lookup_table(
// CHECK-NEXT:   [[sks:v[0-9]+]]: &ServerKey,
// CHECK-NEXT:   [[lut:v[0-9]+]]: &LookupTableOwned,
// CHECK-NEXT:   [[input:v[0-9]+]]: &Ciphertext,
// CHECK-NEXT: ) -> Ciphertext {
// CHECK-NEXT:   let [[v0:.*]] = [[sks]].apply_lookup_table(&[[input]], &[[lut]]);
// CHECK-NEXT:   [[v0]]
// CHECK-NEXT: }
func.func @test_apply_lookup_table(%sks : !sks, %lut: !lut, %input : !eui3) -> !eui3 {
  %out = tfhe_rust.apply_lookup_table %sks, %input, %lut : (!sks, !eui3, !lut) -> !eui3
  return %out : !eui3
}

// CHECK-LABEL: pub fn test_apply_lookup_table2(
// CHECK-NEXT:   [[sks:v[0-9]+]]: &ServerKey,
// CHECK-NEXT:   [[lut:v[0-9]+]]: &LookupTableOwned,
// CHECK-NEXT:   [[input:v[0-9]+]]: &Ciphertext,
// CHECK-NEXT: ) -> Ciphertext {
// CHECK-NEXT:   let [[v1:.*]] = [[sks]].apply_lookup_table(&[[input]], &[[lut]]);
// CHECK-NEXT:   let [[v2:.*]] = [[sks]].unchecked_add(&[[input]], &[[v1]]);
// CHECK-NEXT:   let [[c1:.*]] = 1;
// CHECK-NEXT:   let [[v3:.*]] = [[sks]].scalar_left_shift(&[[v2]], &[[c1]]);
// CHECK-NEXT:   let [[v4:.*]] = [[sks]].apply_lookup_table(&[[v3]], &[[lut]]);
// CHECK-NEXT:   [[v4]]
// CHECK-NEXT: }
func.func @test_apply_lookup_table2(%sks : !sks, %lut: !lut, %input : !eui3) -> !eui3 {
  %v1 = tfhe_rust.apply_lookup_table %sks, %input, %lut : (!sks, !eui3, !lut) -> !eui3
  %v2 = tfhe_rust.add %sks, %input, %v1 : (!sks, !eui3, !eui3) -> !eui3
  %c1 = arith.constant 1 : i8
  %v3 = tfhe_rust.scalar_left_shift %sks, %v2, %c1 : (!sks, !eui3, i8) -> !eui3
  %v4 = tfhe_rust.apply_lookup_table %sks, %v3, %lut : (!sks, !eui3, !lut) -> !eui3
  return %v4 : !eui3
}

// CHECK-LABEL: pub fn test_return_multiple_values(
// CHECK-NEXT:   [[input:v[0-9]+]]: &Ciphertext,
// CHECK-NEXT: ) -> (Ciphertext, Ciphertext) {
// CHECK-NEXT:   ([[input]], [[input]])
// CHECK-NEXT: }
func.func @test_return_multiple_values(%input : !eui3) -> (!eui3, !eui3) {
  return %input, %input : !eui3, !eui3
}
