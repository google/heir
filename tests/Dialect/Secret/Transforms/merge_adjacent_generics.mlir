// RUN: heir-opt --secret-merge-adjacent-generics --split-input-file %s | FileCheck %s

// CHECK: test_chained_input_output
// CHECK-SAME: (%[[value:.*]]: !secret.secret<i32>
!ty = !secret.secret<i32>
func.func @test_chained_input_output(%value : !ty) -> !ty {
  // CHECK: %[[c1:.*]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  // CHECK:      secret.generic(%[[value]]
  // CHECK-NEXT: ^body(%[[clear_value:.*]]: i32):
  // CHECK-NEXT:   %[[res:.*]] = arith.addi %[[clear_value]], %[[c1]]
  // CHECK-NEXT:   %[[res2:.*]] = arith.addi %[[res]], %[[c1]]
  // CHECK-NEXT:   secret.yield %[[res2:.*]]
  // CHECK-NOT:   secret.generic
  %0 = secret.generic(%value : !ty) {
    ^body(%x: i32) :
      %res = arith.addi %x, %c1: i32
      secret.yield %res : i32
    } -> (!ty)
  %1 = secret.generic(%0 : !ty) {
    ^body(%x: i32) :
      %res = arith.addi %x, %c1: i32
      secret.yield %res : i32
    } -> (!ty)
  func.return %1 : !ty
}

// -----

// CHECK: test_shared_input
// CHECK-SAME: (%[[v1:.*]]: [[si32:!secret.secret<i32>]], %[[v2:.*]]: !secret.secret<i32>, %[[v3:.*]]: !secret.secret<i32>
!ty = !secret.secret<i32>
func.func @test_shared_input(%v1: !ty, %v2: !ty, %v3: !ty) -> !ty {
  // CHECK:      secret.generic(%[[v1]]: [[si32]], %[[v2]]: [[si32]], %[[v3]]: [[si32]]
  // CHECK-NEXT: ^body(%[[cv1:.*]]: i32, %[[cv2:.*]]: i32, %[[cv3:.*]]: i32
  // CHECK-NEXT:   %[[r1:.*]] = arith.addi %[[cv1]], %[[cv2]]
  // CHECK-NEXT:   %[[r2:.*]] = arith.addi %[[cv2]], %[[cv3]]
  // CHECK-NEXT:   secret.yield %[[r1]], %[[r2]]
  // CHECK-NOT:  secret.generic
  %0 = secret.generic(%v1: !ty, %v2: !ty) {
    ^body(%clear1: i32, %clear2: i32) :
      %res = arith.addi %clear1, %clear2: i32
      secret.yield %res : i32
    } -> (!ty)
  %1 = secret.generic(%v2: !ty,  %v3: !ty) {
    ^body(%clear2: i32, %clear3: i32) :
      %res = arith.addi %clear2, %clear3: i32
      secret.yield %res : i32
    } -> (!ty)
  func.return %1 : !ty
}

// -----

// CHECK: test_unshared_input
// CHECK-SAME: (%[[v1:.*]]: [[si32:!secret.secret<i32>]], %[[v2:.*]]: !secret.secret<i32>, %[[v3:.*]]: !secret.secret<i32>
!ty = !secret.secret<i32>
func.func @test_unshared_input(%v1: !ty, %v2: !ty, %v3: !ty) -> !ty {
  // CHECK: %[[c1:.*]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  // CHECK:      secret.generic(%[[v1]]: [[si32]], %[[v2]]: [[si32]], %[[v3]]: [[si32]]
  // CHECK-NEXT: ^body(%[[cv1:.*]]: i32, %[[cv2:.*]]: i32, %[[cv3:.*]]: i32
  // CHECK-NEXT:   %[[r1:.*]] = arith.addi %[[cv1]], %[[c1]]
  // CHECK-NEXT:   %[[r2:.*]] = arith.addi %[[cv2]], %[[cv3]]
  // CHECK-NEXT:   secret.yield %[[r1]], %[[r2]]
  // CHECK-NOT:  secret.generic
  %0 = secret.generic(%v1 : !ty) {
    ^body(%clear1: i32) :
      %res = arith.addi %clear1, %c1: i32
      secret.yield %res : i32
    } -> (!ty)
  %1 = secret.generic(%v2: !ty, %v3: !ty) {
    ^body(%clear2: i32, %clear3: i32) :
      %res = arith.addi %clear2, %clear3: i32
      secret.yield %res : i32
    } -> (!ty)
  func.return %1 : !ty
}
