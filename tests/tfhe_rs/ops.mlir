// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!sks = !tfhe_rs.server_key
module {
  // CHECK-LABEL: func @test_create_trivial
  func.func @test_create_trivial(%sks : !sks) {
    %0 = arith.constant 1 : i8
    %1 = arith.constant 1 : i3
    %2 = arith.constant 1 : i128
    %e1 = tfhe_rs.create_trivial %sks, %0 : (!sks, i8) -> !tfhe_rs.ei8
    %eu1 = tfhe_rs.create_trivial %sks, %1 : (!sks, i3) -> !tfhe_rs.eui8
    %e2 = tfhe_rs.create_trivial %sks, %2 : (!sks, i128) -> !tfhe_rs.ei128
    return
  }
}
