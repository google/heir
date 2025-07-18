// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

module {
  // CHECK: func @fooFunc
  func.func @fooFunc(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
    return %arg0 : !secret.secret<i32>
  }

  // CHECK: noInputs
  func.func @noInputs() -> !secret.secret<memref<1x16xi8>> {
    // CHECK: secret.generic
    %Z = secret.generic() {
        %d = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
        secret.yield %d : memref<1x16xi8>
      } -> (!secret.secret<memref<1x16xi8>>)
    func.return %Z : !secret.secret<memref<1x16xi8>>
  }

  // CHECK: conceal_trivial
  func.func @conceal_trivial() -> !secret.secret<i8> {
    %c7 = arith.constant 7 : i8
    %0 = secret.conceal %c7 {trivial} : i8 -> !secret.secret<i8>
    func.return %0 : !secret.secret<i8>
  }
}
