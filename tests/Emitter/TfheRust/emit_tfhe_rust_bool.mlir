// RUN: heir-translate %s --emit-tfhe-rust-hl | FileCheck %s

module {
  // CHECK: add_round_key
  // CHECK-NEXT: [[v1:.*]]: &[Ciphertext; 2],
  // CHECK-NEXT: [[v2:.*]]: &[Ciphertext; 2]
  // CHECK-DAG: let [[v4:.*]] = 0usize;
  // CHECK-DAG: let [[c1:.*]] = 1usize;
  // CHECK: let mut [[v5:.*]]: BTreeMap
  // CHECK: let [[v8:.*]] = [[v6:.*]] ^ [[v7:.*]];
  func.func @add_round_key(%arg0: !tfhe_rust.server_key, %arg1: memref<2x!tfhe_rust.eui8> {secret.secret}, %arg2: memref<2x!tfhe_rust.eui8> {secret.secret}) -> memref<2x!tfhe_rust.eui8> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x!tfhe_rust.eui8>
    %0 = memref.load %arg1[%c0] : memref<2x!tfhe_rust.eui8>
    %1 = memref.load %arg2[%c0] : memref<2x!tfhe_rust.eui8>
    %2 = tfhe_rust.bitxor %arg0, %0, %1 : (!tfhe_rust.server_key, !tfhe_rust.eui8, !tfhe_rust.eui8) -> !tfhe_rust.eui8
    memref.store %2, %alloc[%c0] {lwe_annotation = "LWE"} : memref<2x!tfhe_rust.eui8>
    %3 = memref.load %arg1[%c1] : memref<2x!tfhe_rust.eui8>
    %4 = memref.load %arg2[%c1] : memref<2x!tfhe_rust.eui8>
    %5 = tfhe_rust.bitxor %arg0, %3, %4 : (!tfhe_rust.server_key, !tfhe_rust.eui8, !tfhe_rust.eui8) -> !tfhe_rust.eui8
    memref.store %5, %alloc[%c1] {lwe_annotation = "LWE"} : memref<2x!tfhe_rust.eui8>
    return %alloc : memref<2x!tfhe_rust.eui8>
  }
}
