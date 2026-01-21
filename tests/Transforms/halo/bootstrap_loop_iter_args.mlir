// RUN: heir-opt --bootstrap-loop-iter-args %s | FileCheck %s

// CHECK: @doctest([[arg0:%[^:]*]]: !secret.secret<i32>)
// CHECK:   [[c1:%[^ ]*]] = arith.constant 1 : i32
// CHECK:   [[res:%[^ ]*]] = secret.generic([[arg0]]
// CHECK:   ^[[body:[^(]*]]([[arg0_val:%[^:]*]]: i32):
// CHECK:     [[peeled:%[^ ]*]] = arith.addi [[c1]], [[arg0_val]]
// CHECK:     [[level_reduced:%[^ ]*]] = mgmt.level_reduce_min [[peeled]]
// CHECK:     [[res:%[^ ]*]] = affine.for [[i:[^ ]*]] = 1 to 10 iter_args([[iter_arg:%[^ ]*]] = [[level_reduced]]) -> (i32) {
// CHECK:       mgmt.bootstrap [[iter_arg]]
// CHECK:       arith.addi
// CHECK:       mgmt.level_reduce_min
// CHECK:       affine.yield
func.func @doctest(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c1 = arith.constant 1 : i32
  %0 = secret.generic(%arg0: !secret.secret<i32>) {
  ^body(%arg0_val: i32):
    %init = arith.addi %c1, %arg0_val : i32
    %res = affine.for %i = 1 to 10 iter_args(%sum_iter = %init) -> i32 {
      %sum = arith.addi %sum_iter, %arg0_val : i32
      affine.yield %sum : i32
    }
    secret.yield %res : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}

// CHECK: @doctest_scf([[arg0:%[^:]*]]: !secret.secret<i32>)
// CHECK:   [[c1:%[^ ]*]] = arith.constant 1 : i32
// CHECK:   [[res:%[^ ]*]] = secret.generic([[arg0]]
// CHECK:   ^[[body:[^(]*]]([[arg0_val:%[^:]*]]: i32):
// CHECK:     [[peeled:%[^ ]*]] = arith.addi [[c1]], [[arg0_val]]
// CHECK:     [[level_reduced:%[^ ]*]] = mgmt.level_reduce_min [[peeled]]
// CHECK:     [[res:%[^ ]*]] = scf.for [[i:[^ ]*]] = %c1 to %c10 step %c1 iter_args([[iter_arg:%[^ ]*]] = [[level_reduced]]) -> (i32) {
// CHECK:       mgmt.bootstrap [[iter_arg]]
// CHECK:       arith.addi
// CHECK:       mgmt.level_reduce_min
// CHECK:       scf.yield
func.func @doctest_scf(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = secret.generic(%arg0: !secret.secret<i32>) {
  ^body(%arg0_val: i32):
    %init = arith.addi %c1_i32, %arg0_val : i32
    %res = scf.for %i = %c1 to %c10 step %c1 iter_args(%sum_iter = %init) -> i32 {
      %sum = arith.addi %sum_iter, %arg0_val : i32
      scf.yield %sum : i32
    }
    secret.yield %res : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}

// CHECK: @four_iter_args_two_secret
// CHECK:         [[C0:%[^ ]*]] = arith.constant 0
// CHECK:         [[C1:%[^ ]*]] = arith.constant 1
// CHECK:         secret.generic
// CHECK:         ^body([[S0:%[^:]*]]: i32, [[S1:%[^:]*]]: i32):
// CHECK:           [[INIT_S0:%[^ ]*]] = arith.addi [[S0]], [[S0]]
// CHECK:           [[INIT_S1:%[^ ]*]] = arith.addi [[S1]], [[S1]]
// CHECK:           [[LRM0:%[^ ]*]] = mgmt.level_reduce_min [[INIT_S0]]
// CHECK:           [[LRM1:%[^ ]*]] = mgmt.level_reduce_min [[INIT_S1]]
// CHECK:           affine.for {{.*}} iter_args([[ARG0:%[^ ]*]] = [[C0]], [[ARG1:%[^ ]*]] = [[LRM0]], [[ARG2:%[^ ]*]] = [[C1]], [[ARG3:%[^ ]*]] = [[LRM1]])
// CHECK:             [[BS0:%[^ ]*]] = mgmt.bootstrap [[ARG1]]
// CHECK:             [[BS1:%[^ ]*]] = mgmt.bootstrap [[ARG3]]
// CHECK:             [[ADD0:%[^ ]*]] = arith.addi [[ARG0]], [[C0]]
// CHECK:             [[ADD1:%[^ ]*]] = arith.addi [[BS0]], [[S0]]
// CHECK:             [[ADD2:%[^ ]*]] = arith.addi [[ARG2]], [[C1]]
// CHECK:             [[ADD3:%[^ ]*]] = arith.addi [[BS1]], [[S1]]
// CHECK:             [[LRM2:%[^ ]*]] = mgmt.level_reduce_min [[ADD1]]
// CHECK:             [[LRM3:%[^ ]*]] = mgmt.level_reduce_min [[ADD3]]
// CHECK:             affine.yield [[ADD0]], [[LRM2]], [[ADD2]], [[LRM3]]
// CHECK:           secret.yield
func.func @four_iter_args_two_secret(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> (!secret.secret<i32>, !secret.secret<i32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %0:2 = secret.generic(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) {
  ^body(%s0: i32, %s1: i32):
    %init_s0 = arith.addi %s0, %s0 : i32
    %init_s1 = arith.addi %s1, %s1 : i32
    %res:4 = affine.for %i = 1 to 10 iter_args(%iter0 = %c0, %iter1 = %init_s0, %iter2 = %c1, %iter3 = %init_s1) -> (i32, i32, i32, i32) {
      %add0 = arith.addi %iter0, %c0 : i32
      %add1 = arith.addi %iter1, %s0 : i32
      %add2 = arith.addi %iter2, %c1 : i32
      %add3 = arith.addi %iter3, %s1 : i32
      affine.yield %add0, %add1, %add2, %add3 : i32, i32, i32, i32
    }
    secret.yield %res#1, %res#3 : i32, i32
  } -> (!secret.secret<i32>, !secret.secret<i32>)
  return %0#0, %0#1 : !secret.secret<i32>, !secret.secret<i32>
}
