// RUN: heir-opt --secret-capture-generic-ambient-scope %s | FileCheck %s

// CHECK: test_capture_ambient_scope
func.func @test_capture_ambient_scope(%value : i32) {
  // CHECK: %[[X:.*]] = arith.constant 7
  %X = arith.constant 7 : i32
  // CHECK: %[[Y:.*]] = secret.conceal
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  // CHECK: %[[Z:.*]] = secret.generic(%[[Y]]: !secret.secret<i32>, %[[X]]: i32)
  %Z = secret.generic(%Y : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[y:.*]]: i32, %[[x:.*]]: i32):
  ^bb0(%y: i32):
    // CHECK-NEXT: %[[d:.*]] = arith.addi %[[x]], %[[y]]
    %d = arith.addi %X, %y : i32
    secret.yield %d : i32
  } -> (!secret.secret<i32>)
  func.return
}

// Regression test for capturing the scope of an operand that was used within a
// child region of the generic. The code would not update the operand with the
// newly captured block argument since the operation's parent was different than
// (but strictly contained in) the generic, causing an infinite loop of added
// block arguments for the operand.
// CHECK: test_capture_within_region
func.func @test_capture_within_region(%value : i32) {
  // CHECK: %[[X:.*]] = arith.constant 7
  %X = arith.constant 7 : i32
  // CHECK: %[[Y:.*]] = secret.conceal
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  // CHECK: %[[Z:.*]] = secret.generic(%[[Y]]: !secret.secret<i32>, %[[X]]: i32)
  %Z = secret.generic(%Y : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[y:.*]]: i32, %[[x:.*]]: i32):
  ^bb0(%y: i32):
    %sum = affine.for %i = 0 to 1 iter_args(%sum_iter = %X) -> (i32) {
      %d = arith.addi %X, %sum_iter : i32
      affine.yield %d : i32
    }
    secret.yield %sum : i32
  } -> (!secret.secret<i32>)
  func.return
}
