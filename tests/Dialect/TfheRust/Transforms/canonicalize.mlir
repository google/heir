// RUN: heir-opt --canonicalize %s | FileCheck %s

!sks = !tfhe_rust.server_key

module {
  // CHECK-LABEL: func @test_move_create_trivial
  func.func @test_move_create_trivial(%sks : !sks, %lut: !tfhe_rust.lookup_table) -> !tfhe_rust.eui3 {
    // CHECK: arith.constant
    // CHECK-NEXT: tfhe_rust.create_trivial
    // CHECK-NEXT: tfhe_rust.create_trivial
    %0 = arith.constant 1 : i3
    %e2 = tfhe_rust.create_trivial %sks, %0 : (!sks, i3) -> !tfhe_rust.eui3
    %e2Shifted = tfhe_rust.scalar_left_shift %sks, %e2 {shiftAmount = 1 : index} : (!sks, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %e1 = tfhe_rust.create_trivial %sks, %0 : (!sks, i3) -> !tfhe_rust.eui3
    %eCombined =  tfhe_rust.add %sks, %e1, %e2Shifted : (!sks, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %out = tfhe_rust.apply_lookup_table %sks, %eCombined, %lut : (!sks, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    return %out : !tfhe_rust.eui3
  }

  // CHECK-LABEL: func @test_move_out_of_loop
  func.func @test_move_out_of_loop(%sks : !sks, %lut: !tfhe_rust.lookup_table) -> memref<10x!tfhe_rust.eui3> {
    // CHECK: arith.constant
    // CHECK-NEXT: tfhe_rust.create_trivial
    // CHECK-NEXT: tfhe_rust.create_trivial
    // CHECK-NEXT: memref.alloc
    // CHECK-NEXT: affine.for
    %0 = arith.constant 1 : i3
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

  // CHECK-LABEL: func @test_move_to_front_of_block
  func.func @test_move_to_front_of_block(%sks : !sks, %value : i3) -> (!tfhe_rust.eui3, i3, i3) {
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: tfhe_rust.create_trivial
    // CHECK-NEXT: tfhe_rust.create_trivial
    // CHECK-NEXT: arith.addi
    %c3 = arith.constant 3 : i3
    %c2 = arith.constant 2 : i3
    %c1 = arith.constant 1 : i3
    %sum = arith.addi %c3, %value : i3
    %enc_val_2 = tfhe_rust.create_trivial %sks, %c1 : (!sks, i3) -> !tfhe_rust.eui3
    %enc_val = tfhe_rust.create_trivial %sks, %value : (!sks, i3) -> !tfhe_rust.eui3
    %combined = tfhe_rust.add %sks, %enc_val, %enc_val_2 : (!sks, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    return %combined, %c2, %sum : !tfhe_rust.eui3, i3, i3
  }
}
