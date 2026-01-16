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
