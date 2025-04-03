// RUN: heir-translate %s --emit-tfhe-rust --use-levels=True | FileCheck %s

!sks = !tfhe_rust.server_key

!lut = !tfhe_rust.lookup_table
!eui3 = !tfhe_rust.eui3

// CHECK: pub fn test_levelled_op(
// CHECK: ) -> Ciphertext {
// CHECK-COUNT-4: static LEVEL_
// CHECK-NOT: static LEVEL_
// CHECK-COUNT-4:   run_level
// CHECK-NOT: run_level
// CHECK:  temp_nodes[
// CHECK-NEXT: }
func.func @test_levelled_op(%sks : !sks, %lut: !lut, %input1 : !eui3, %input2 : !eui3) -> !eui3 {
  %v0 = tfhe_rust.apply_lookup_table %sks, %input1, %lut : (!sks, !eui3, !lut) -> !eui3
  %v1 = tfhe_rust.apply_lookup_table %sks, %input2, %lut : (!sks, !eui3, !lut) -> !eui3
  %v2 = tfhe_rust.add %sks, %v0, %v1 : (!sks, !eui3, !eui3) -> !eui3
  %v3 = tfhe_rust.scalar_left_shift %sks, %v2 {shiftAmount = 1 : index} : (!sks, !eui3) -> !eui3
  %v4 = tfhe_rust.apply_lookup_table %sks, %v3, %lut : (!sks, !eui3, !lut) -> !eui3
  return %v4 : !eui3
}

// CHECK: pub fn test_levelled_op_break(
// CHECK: ) -> Ciphertext {
// CHECK-COUNT-2: static LEVEL_
// CHECK-NOT: static LEVEL_
// CHECK-COUNT-2:   run_level
// CHECK-NOT: run_level
// CHECK: let [[v0:.*]] = 1;
// CHECK-COUNT-2: static LEVEL_
// CHECK-NOT: static LEVEL_
// CHECK-COUNT-2:   run_level
// CHECK-NOT: run_level
// CHECK:  temp_nodes[
// CHECK-NEXT: }

// This tests a non-levelled op interleaved between to segments of levelled ops.
// Typically, these will be affine.for statements, or memref.load or stores.
func.func @test_levelled_op_break(%sks : !sks, %lut: !lut, %input1 : !eui3, %input2 : !eui3) -> !eui3 {
  %v0 = tfhe_rust.apply_lookup_table %sks, %input1, %lut : (!sks, !eui3, !lut) -> !eui3
  %v1 = tfhe_rust.apply_lookup_table %sks, %input2, %lut : (!sks, !eui3, !lut) -> !eui3
  %v2 = tfhe_rust.add %sks, %v0, %v1 : (!sks, !eui3, !eui3) -> !eui3
  %c1 = arith.constant 1 : i8
  %v3 = tfhe_rust.scalar_left_shift %sks, %v2 {shiftAmount = 1 : index} : (!sks, !eui3) -> !eui3
  %v4 = tfhe_rust.apply_lookup_table %sks, %v3, %lut : (!sks, !eui3, !lut) -> !eui3
  return %v4 : !eui3
}
