// RUN: heir-opt --secret-distribute-generic %s | FileCheck %s

// CHECK-LABEL: test_distribute_generic
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>, %[[cond:.*]]: i1) -> !secret.secret<i32> {
func.func @test_distribute_generic(%value: !secret.secret<i32>, %cond: i1) -> !secret.secret<i32> {
  // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : i32
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32

  // CHECK-NEXT: %[[g0:.*]] = secret.generic ins(%[[value]], %[[c7]] : !secret.secret<i32>, i32) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[clear_g0_in0:.*]]: i32, %[[clear_g0_in1:.*]]: i32):
  // CHECK-NEXT:   %[[g0_op:.*]] = arith.muli %[[clear_g0_in0]], %[[clear_g0_in1]] : i32
  // CHECK-NEXT:   secret.yield %[[g0_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: %[[g1:.*]] = secret.generic ins(%[[g0]], %[[c1]] : !secret.secret<i32>, i32) {
  // CHECK-NEXT: ^[[bb1:.*]](%[[clear_g1_in0:.*]]: i32, %[[clear_g1_in1:.*]]: i32):
  // CHECK-NEXT:   %[[g1_op:.*]] = arith.addi %[[clear_g1_in0]], %[[clear_g1_in1]] : i32
  // CHECK-NEXT:   secret.yield %[[g1_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: %[[g2:.*]] = secret.generic ins(%[[g1]], %[[g1]] : !secret.secret<i32>, !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb2:.*]](%[[clear_g2_in0:.*]]: i32, %[[clear_g2_in1:.*]]: i32):
  // CHECK-NEXT:   %[[g2_op:.*]] = arith.muli %[[clear_g2_in0]], %[[clear_g2_in1]] : i32
  // CHECK-NEXT:   secret.yield %[[g2_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: %[[g3:.*]] = secret.generic ins(%[[cond]], %[[c0]], %[[g2]] : i1, i32, !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb3:.*]](%[[clear_g3_in0:.*]]: i1, %[[clear_g3_in1:.*]]: i32, %[[clear_g3_in2:.*]]: i32):
  // CHECK-NEXT:   %[[g3_op:.*]] = arith.select %[[clear_g3_in0]], %[[clear_g3_in2]], %[[clear_g3_in1]] : i32
  // CHECK-NEXT:   secret.yield %[[g3_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: return %[[g3]] : !secret.secret<i32>
  %Z = secret.generic
    ins(%value, %cond : !secret.secret<i32>, i1) {
    ^bb0(%clear_value: i32, %clear_cond: i1):
      // computes (7x + 1)^2 if cond, else 0
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c7 = arith.constant 7 : i32
      %0 = arith.muli %clear_value, %c7 : i32
      %1 = arith.addi %0, %c1 : i32
      %2 = arith.muli %1, %1 : i32
      %3 = arith.select %clear_cond, %2, %c0 : i32
      secret.yield %3 : i32
    } -> (!secret.secret<i32>)
  func.return %Z : !secret.secret<i32>
}


// TODO(https://github.com/google/heir/issues/170): test
// scf.if, affine.for.
