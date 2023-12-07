// RUN: heir-opt --secret-capture-generic-ambient-scope %s | FileCheck %s

// CHECK-LABEL: test_capture_ambient_scope
func.func @test_capture_ambient_scope(%value : i32) {
  // CHECK: %[[X:.*]] = arith.constant 7
  %X = arith.constant 7 : i32
  // CHECK: %[[Y:.*]] = secret.conceal
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  // CHECK: %[[Z:.*]] = secret.generic ins(%[[Y]], %[[X]] : !secret.secret<i32>, i32)
  %Z = secret.generic ins(%Y : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[y:.*]]: i32, %[[x:.*]]: i32):
  ^bb0(%y: i32):
    // CHECK-NEXT: %[[d:.*]] = arith.addi %[[x]], %[[y]]
    %d = arith.addi %X, %y : i32
    secret.yield %d : i32
  } -> (!secret.secret<i32>)
  func.return
}
