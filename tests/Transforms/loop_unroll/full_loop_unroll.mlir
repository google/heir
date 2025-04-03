// RUN: heir-opt --full-loop-unroll %s | FileCheck %s

!sks = !tfhe_rust.server_key

// CHECK: func @test_move_out_of_loop
func.func @test_move_out_of_loop(%sks : !sks, %lut: !tfhe_rust.lookup_table) -> memref<10x!tfhe_rust.eui3> {
  // CHECK-NOT: affine.for
  %0 = arith.constant 1 : i3
  %1 = arith.constant 2 : i3
  %memref = memref.alloca() : memref<10x!tfhe_rust.eui3>

  affine.for %i = 0 to 10 {
    %e2 = tfhe_rust.create_trivial %sks, %0 : (!sks, i3) -> !tfhe_rust.eui3
    %e2Shifted = tfhe_rust.scalar_left_shift %sks, %e2 {shiftAmount = 1 : index} : (!sks, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %e1 = tfhe_rust.create_trivial %sks, %0 : (!sks, i3) -> !tfhe_rust.eui3
    %eCombined =  tfhe_rust.add %sks, %e1, %e2Shifted : (!sks, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %out = tfhe_rust.apply_lookup_table %sks, %eCombined, %lut : (!sks, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    memref.store %out, %memref[%i] : memref<10x!tfhe_rust.eui3>
    affine.yield
  }
  return %memref : memref<10x!tfhe_rust.eui3>
}
