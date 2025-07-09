// RUN: heir-translate %s --emit-tfhe-rust --use-levels=False | FileCheck %s

// CHECK: pub fn mix(
// CHECK-NEXT:   [[input1:v[0-9]+]]: &[Ciphertext; 4],
// CHECK-NEXT: ) -> [Ciphertext; 4] {
// CHECK: let mut [[v5:.*]] : HashMap<usize, Ciphertext> = HashMap::new();
// CHECK-COUNT-4: [[v5]].insert
// CHECK: core::array::from_fn
// CHECK-NEXT: }
func.func @mix(%input1 : memref<4x!tfhe_rust.bool>) -> memref<4x!tfhe_rust.bool> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x!tfhe_rust.bool>
  %1 = memref.load %input1[%c0] : memref<4x!tfhe_rust.bool>
  %2 = memref.load %input1[%c1] : memref<4x!tfhe_rust.bool>
  %3 = memref.load %input1[%c2] : memref<4x!tfhe_rust.bool>
  %4 = memref.load %input1[%c3] : memref<4x!tfhe_rust.bool>
  memref.store %2, %alloc[%c0] : memref<4x!tfhe_rust.bool>
  memref.store %1, %alloc[%c1] : memref<4x!tfhe_rust.bool>
  memref.store %3, %alloc[%c2] : memref<4x!tfhe_rust.bool>
  memref.store %4, %alloc[%c3] : memref<4x!tfhe_rust.bool>
  return %alloc : memref<4x!tfhe_rust.bool>
}
