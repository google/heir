// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!bsks = !tfhe_rust_bool.server_key
!eb = !tfhe_rust_bool.eb

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

  // CHECK-LABEL: func @test_packed_and
  func.func @test_packed_and(%bsks : !bsks, %lhs : tensor<4x!eb>, %rhs : tensor<4x!eb>) {
    %0 = arith.constant 0 : index
    %1 = arith.constant 1 : index
    %4 = arith.constant 4 : index

    %c0 = arith.constant 0 : i1
    %c1 = arith.constant 1 : i1

    scf.for %i = %0 to %4 step %1 {
      %tmp1 = tfhe_rust_bool.create_trivial %bsks, %c0 : (!bsks, i1) -> !eb
      %tmp2 = tfhe_rust_bool.create_trivial %bsks, %c1 : (!bsks, i1) -> !eb

      tensor.insert %tmp1 into %lhs[%i] : tensor<4x!eb>
      tensor.insert %tmp2 into %rhs[%i] : tensor<4x!eb>
    }

    %out = tfhe_rust_bool.and_packed %bsks, %lhs, %rhs: (!bsks, tensor<4x!eb>, tensor<4x!eb>) -> tensor<4x!eb>
    return
  }
}


