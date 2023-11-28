// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!sks = !tfhe_rust.server_key
module {
  // CHECK-LABEL: func @test_create_trivial
  func.func @test_create_trivial(%sks : !sks) {
    %0 = arith.constant 1 : i8
    %1 = arith.constant 1 : i3
    %2 = arith.constant 1 : i128
    %e1 = tfhe_rust.create_trivial %sks, %0 : (!sks, i8) -> !tfhe_rust.ei8
    %eu1 = tfhe_rust.create_trivial %sks, %1 : (!sks, i3) -> !tfhe_rust.eui8
    %e2 = tfhe_rust.create_trivial %sks, %2 : (!sks, i128) -> !tfhe_rust.ei128
    return
  }

  // CHECK-LABEL: func @test_apply_lookup_table
  func.func @test_apply_lookup_table(%sks : !sks, %lut: !tfhe_rust.lookup_table) {
    %0 = arith.constant 1 : i3
    %1 = arith.constant 2 : i3
    %e1 = tfhe_rust.create_trivial %sks, %0 : (!sks, i3) -> !tfhe_rust.eui3
    %e2 = tfhe_rust.create_trivial %sks, %0 : (!sks, i3) -> !tfhe_rust.eui3

    %shiftAmount = arith.constant 1 : i8
    %e2Shifted = tfhe_rust.scalar_left_shift %sks, %e2, %shiftAmount : (!sks, !tfhe_rust.eui3, i8) -> !tfhe_rust.eui3
    %eCombined =  tfhe_rust.add %sks, %e1, %e2Shifted : (!sks, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3

    %out = tfhe_rust.apply_lookup_table %sks, %eCombined, %lut : (!sks, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    return
  }
}
