// RUN: heir-opt --secret-distribute-generic %s | FileCheck %s

// CHECK-LABEL: test_distribute_generic
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>, %[[cond:.*]]: i1) -> !secret.secret<i32> {
func.func @test_distribute_generic(%value: !secret.secret<i32>, %cond: i1) -> !secret.secret<i32> {
  // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : i32
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32

  // CHECK-NEXT: %[[g0:.*]] = secret.generic ins(%[[value]] : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[clear_g0_in0:.*]]: i32):
  // CHECK-NEXT:   %[[g0_op:.*]] = arith.muli %[[clear_g0_in0]], %[[c7]] : i32
  // CHECK-NEXT:   secret.yield %[[g0_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: %[[g1:.*]] = secret.generic ins(%[[g0]] : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb1:.*]](%[[clear_g1_in0:.*]]: i32):
  // CHECK-NEXT:   %[[g1_op:.*]] = arith.addi %[[clear_g1_in0]], %[[c1]] : i32
  // CHECK-NEXT:   secret.yield %[[g1_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: %[[g2:.*]] = secret.generic ins(%[[g1]], %[[g1]] : !secret.secret<i32>, !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb2:.*]](%[[clear_g2_in0:.*]]: i32, %[[clear_g2_in1:.*]]: i32):
  // CHECK-NEXT:   %[[g2_op:.*]] = arith.muli %[[clear_g2_in0]], %[[clear_g2_in1]] : i32
  // CHECK-NEXT:   secret.yield %[[g2_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: %[[g3:.*]] = secret.generic ins(%[[g2]] : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb3:.*]](%[[clear_g3_in2:.*]]: i32):
  // CHECK-NEXT:   %[[g3_op:.*]] = arith.select %[[cond]], %[[clear_g3_in2]], %[[c0]] : i32
  // CHECK-NEXT:   secret.yield %[[g3_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: return %[[g3]] : !secret.secret<i32>
  %Z = secret.generic
    ins(%value : !secret.secret<i32>) {
    ^bb0(%clear_value: i32):
      // computes (7x + 1)^2 if cond, else 0
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c7 = arith.constant 7 : i32
      %0 = arith.muli %clear_value, %c7 : i32
      %1 = arith.addi %0, %c1 : i32
      %2 = arith.muli %1, %1 : i32
      %3 = arith.select %cond, %2, %c0 : i32
      secret.yield %3 : i32
    } -> (!secret.secret<i32>)
  func.return %Z : !secret.secret<i32>
}


// CHECK-LABEL: test_scf_for
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>) -> !secret.secret<i32> {
func.func @test_scf_for(%value: !secret.secret<i32>) -> !secret.secret<i32> {
  // CHECK-DAG: %[[c1i32:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[c5:.*]] = arith.constant 5 : index

  // The for loop takes the secret as iter_arg and returns it
  // CHECK-NEXT: %[[v0:.*]] = scf.for %[[i:.*]] = %[[c1]] to %[[c5]] step %[[c1]]
  // CHECK-SAME: iter_args(%[[iter_arg:.*]] = %[[value]])
  // CHECK-SAME: -> (!secret.secret<i32>) {

  // The loop body is a single secret.generic for the add
  // CHECK-NEXT: %[[g0:.*]] = secret.generic ins(%[[iter_arg]] : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[clear_iter_arg:.*]]: i32):
  // CHECK-NEXT:   %[[g0_op:.*]] = arith.addi %[[clear_iter_arg]], %[[c1i32]] : i32
  // CHECK-NEXT:   secret.yield %[[g0_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // Terminators are not part of a generic.
  // CHECK-NEXT: scf.yield %[[g0]] : !secret.secret<i32>
  // CHECK-NEXT: }

  // CHECK-NEXT: return %[[v0]]
  %Z = secret.generic
    ins(%value : !secret.secret<i32>) {
    ^bb0(%clear_value: i32):
      %0 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c5 = arith.constant 5 : index
      %1 = scf.for %i = %c1 to %c5 step %c1 iter_args(%iter_arg = %clear_value) -> i32 {
        %2 = arith.addi %0, %iter_arg : i32
        scf.yield %2 : i32
      }
      secret.yield %1 : i32
    } -> (!secret.secret<i32>)
  func.return %Z : !secret.secret<i32>
}

// CHECK-LABEL: test_affine_for
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>
// CHECK-SAME: %[[data:.*]]: !secret.secret<memref<10xi32>>
func.func @test_affine_for(
    %value: !secret.secret<i32>,
    %data: !secret.secret<memref<10xi32>>) -> !secret.secret<memref<10xi32>> {
  // CHECK: affine.for
  // CHECK: secret.generic
  // CHECK:   affine.load
  // CHECK: secret.generic
  // CHECK:   arith.addi
  // CHECK: secret.generic
  // CHECK:   affine.store
  // CHECK: return %[[data]]
  secret.generic
    ins(%value, %data : !secret.secret<i32>, !secret.secret<memref<10xi32>>) {
    ^bb0(%clear_value: i32, %clear_data : memref<10xi32>):
      affine.for %i = 0 to 10 {
        %2 = affine.load %clear_data[%i] : memref<10xi32>
        %3 = arith.addi %2, %clear_value : i32
        affine.store %3, %clear_data[%i] : memref<10xi32>
      }
      secret.yield
    } -> ()
  func.return %data : !secret.secret<memref<10xi32>>
}
