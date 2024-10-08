// RUN: heir-opt --secret-merge-adjacent-generics --split-input-file %s | FileCheck %s

// CHECK-LABEL: test_chained_input_output
// CHECK-SAME: (%[[value:.*]]: !secret.secret<i32>
!ty = !secret.secret<i32>
func.func @test_chained_input_output(%value : !ty) -> !ty {
  // CHECK: %[[c1:.*]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  // CHECK:      secret.generic ins(%[[value]]
  // CHECK-NEXT: ^bb{{[0-9]}}(%[[clear_value:.*]]: i32):
  // CHECK-NEXT:   %[[res:.*]] = arith.addi %[[clear_value]], %[[c1]]
  // CHECK-NEXT:   %[[res2:.*]] = arith.addi %[[res]], %[[c1]]
  // CHECK-NEXT:   secret.yield %[[res2:.*]]
  // CHECK-NOT:   secret.generic
  %0 = secret.generic ins(%value : !ty) {
    ^bb0(%x: i32) :
      %res = arith.addi %x, %c1: i32
      secret.yield %res : i32
    } -> (!ty)
  %1 = secret.generic ins(%0 : !ty) {
    ^bb0(%x: i32) :
      %res = arith.addi %x, %c1: i32
      secret.yield %res : i32
    } -> (!ty)
  func.return %1 : !ty
}

// -----

// CHECK-LABEL: test_shared_input
// CHECK-SAME: (%[[v1:.*]]: !secret.secret<i32>, %[[v2:.*]]: !secret.secret<i32>, %[[v3:.*]]: !secret.secret<i32>
!ty = !secret.secret<i32>
func.func @test_shared_input(%v1: !ty, %v2: !ty, %v3: !ty) -> !ty {
  // CHECK:      secret.generic ins(%[[v1]], %[[v2]], %[[v3]]
  // CHECK-NEXT: ^bb{{[0-9]}}(%[[cv1:.*]]: i32, %[[cv2:.*]]: i32, %[[cv3:.*]]: i32
  // CHECK-NEXT:   %[[r1:.*]] = arith.addi %[[cv1]], %[[cv2]]
  // CHECK-NEXT:   %[[r2:.*]] = arith.addi %[[cv2]], %[[cv3]]
  // CHECK-NEXT:   secret.yield %[[r1]], %[[r2]]
  // CHECK-NOT:  secret.generic
  %0 = secret.generic ins(%v1, %v2 : !ty, !ty) {
    ^bb0(%clear1: i32, %clear2: i32) :
      %res = arith.addi %clear1, %clear2: i32
      secret.yield %res : i32
    } -> (!ty)
  %1 = secret.generic ins(%v2, %v3 : !ty, !ty) {
    ^bb0(%clear2: i32, %clear3: i32) :
      %res = arith.addi %clear2, %clear3: i32
      secret.yield %res : i32
    } -> (!ty)
  func.return %1 : !ty
}

// -----

// CHECK-LABEL: test_unshared_input
// CHECK-SAME: (%[[v1:.*]]: !secret.secret<i32>, %[[v2:.*]]: !secret.secret<i32>, %[[v3:.*]]: !secret.secret<i32>
!ty = !secret.secret<i32>
func.func @test_unshared_input(%v1: !ty, %v2: !ty, %v3: !ty) -> !ty {
  // CHECK: %[[c1:.*]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  // CHECK:      secret.generic ins(%[[v1]], %[[v2]], %[[v3]]
  // CHECK-NEXT: ^bb{{[0-9]}}(%[[cv1:.*]]: i32, %[[cv2:.*]]: i32, %[[cv3:.*]]: i32
  // CHECK-NEXT:   %[[r1:.*]] = arith.addi %[[cv1]], %[[c1]]
  // CHECK-NEXT:   %[[r2:.*]] = arith.addi %[[cv2]], %[[cv3]]
  // CHECK-NEXT:   secret.yield %[[r1]], %[[r2]]
  // CHECK-NOT:  secret.generic
  %0 = secret.generic ins(%v1 : !ty) {
    ^bb0(%clear1: i32) :
      %res = arith.addi %clear1, %c1: i32
      secret.yield %res : i32
    } -> (!ty)
  %1 = secret.generic ins(%v2, %v3 : !ty, !ty) {
    ^bb0(%clear2: i32, %clear3: i32) :
      %res = arith.addi %clear2, %clear3: i32
      secret.yield %res : i32
    } -> (!ty)
  func.return %1 : !ty
}
