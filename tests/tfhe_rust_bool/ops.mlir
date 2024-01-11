// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!bsks = !tfhe_rust_bool.server_key

module {
  // CHECK-LABEL: func @test_create_trivial_bool
  func.func @test_create_trivial_bool(%bsks : !bsks) {
    %0 = arith.constant 1 : i1
    %1 = arith.constant 0 : i1

    %e1 = tfhe_rust_bool.create_trivial %bsks, %0 : (!bsks, i1) -> !tfhe_rust_bool.eb
    %e2 = tfhe_rust_bool.create_trivial %bsks, %1 : (!bsks, i1) -> !tfhe_rust_bool.eb
    return
  }

  // CHECK-LABEL: func @test_and
  func.func @test_and(%bsks : !bsks) {
    %0 = arith.constant 1 : i1
    %1 = arith.constant 1 : i1

    %e1 = tfhe_rust_bool.create_trivial %bsks, %0 : (!bsks, i1) -> !tfhe_rust_bool.eb
    %e2 = tfhe_rust_bool.create_trivial %bsks, %1 : (!bsks, i1) -> !tfhe_rust_bool.eb
    %out = tfhe_rust_bool.and %bsks, %e1, %e2: (!bsks, !tfhe_rust_bool.eb, !tfhe_rust_bool.eb) -> !tfhe_rust_bool.eb
    return
  }

}
