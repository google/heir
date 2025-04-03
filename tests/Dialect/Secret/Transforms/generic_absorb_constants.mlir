// RUN: heir-opt --secret-generic-absorb-constants %s | FileCheck %s

// CHECK: test_copy_single_constant
// CHECK-SAME: %[[Y:.*]]: !secret.secret<i32>) {
func.func @test_copy_single_constant(%value : !secret.secret<i32>) {
  // CHECK: %[[X:.*]] = arith.constant 7
  %X = arith.constant 7 : i32
  // CHECK: %[[Z:.*]] = secret.generic ins(%[[Y]] : !secret.secret<i32>)
  %Z = secret.generic ins(%value : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[y:.*]]: i32):
  ^bb0(%y: i32):
    // CHECK-NEXT: %[[x:.*]] = arith.constant 7
    // CHECK-NEXT: %[[d:.*]] = arith.addi %[[x]], %[[y]]
    %d = arith.addi %X, %y : i32
    secret.yield %d : i32
  } -> (!secret.secret<i32>)
  func.return
}

// CHECK: test_copy_two_constant
// CHECK-SAME: %[[Y:.*]]: !secret.secret<i32>) {
func.func @test_copy_two_constant(%value : !secret.secret<i32>) {
  // CHECK: %[[C0:.*]] = arith.constant 7
  %C7 = arith.constant 7 : i32
  // CHECK: %[[C1:.*]] = arith.constant 9
  %C9 = arith.constant 9 : i32
  // CHECK: %[[Z:.*]] = secret.generic ins(%[[Y]] : !secret.secret<i32>)
  %Z = secret.generic ins(%value : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[y:.*]]: i32):
  ^bb0(%y: i32):
    // CHECK-NEXT: %[[c0:.*]] = arith.constant 7
    // CHECK-NEXT: %[[c1:.*]] = arith.constant 9
    // CHECK-NEXT: %[[d:.*]] = arith.addi %[[c0]], %[[y]]
    // CHECK-NEXT: %[[e:.*]] = arith.muli %[[c1]], %[[d]]
    %d = arith.addi %C7, %y : i32
    %e = arith.muli %C9, %d : i32
    secret.yield %e : i32
  } -> (!secret.secret<i32>)
  func.return
}

// CHECK: test_block_arg
// CHECK-SAME: %[[Y:.*]]: !secret.secret<i32>) {
func.func @test_block_arg(%value : !secret.secret<i32>) {
  // CHECK: %[[C0:.*]] = arith.constant 7
  %C7 = arith.constant 7 : i32
  // CHECK: %[[Z:.*]] = secret.generic ins(%[[Y]] : !secret.secret<i32>)
  %Z = secret.generic ins(%value, %C7 : !secret.secret<i32>, i32) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[y:.*]]: i32):
  ^bb0(%y: i32, %c0 : i32):
    // CHECK-NEXT: %[[c0:.*]] = arith.constant 7
    // CHECK-NEXT: %[[d:.*]] = arith.addi %[[c0]], %[[y]]
    %d = arith.addi %c0, %y : i32
    secret.yield %d : i32
  } -> (!secret.secret<i32>)
  func.return
}
