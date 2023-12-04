// RUN: heir-opt %s | FileCheck %s

// CHECK-LABEL: test_add_secret_and_plaintext_passed_through_block
func.func @test_add_secret_and_plaintext_passed_through_block(%value : i32) {
  %X = arith.constant 7 : i32
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  // CHECK: secret.generic
  %Z = secret.generic
    ins(%X, %Y : i32, !secret.secret<i32>) {
    ^bb0(%x: i32, %y: i32) :
      %d = arith.addi %x, %y: i32
      secret.yield %d : i32
    } -> (!secret.secret<i32>)
  func.return
}

// CHECK-LABEL: test_add_secret_and_plaintext_in_enclosing_scope
func.func @test_add_secret_and_plaintext_in_enclosing_scope(%value : i32) {
  %X = arith.constant 7 : i32
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  // CHECK: secret.generic
  %Z = secret.generic
    ins(%Y : !secret.secret<i32>) {
    ^bb0(%y: i32) :
      %d = arith.addi %X, %y: i32
      secret.yield %d : i32
    } -> (!secret.secret<i32>)
  func.return
}
