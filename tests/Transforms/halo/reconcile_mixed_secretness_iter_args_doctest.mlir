// RUN: heir-opt --reconcile-mixed-secretness-iter-args %s | FileCheck %s

// CHECK: @peel_first_iter([[arg0:%[^:]*]]: !secret.secret<i32>)
// CHECK:   [[c1:%[^ ]*]] = arith.constant 1 : i32
// CHECK:   [[res:%[^ ]*]] = secret.generic([[arg0]]
// CHECK:   ^[[body:[^(]*]]([[arg0_val:%[^:]*]]: i32):
// CHECK:     [[peeled:%[^ ]*]] = arith.addi [[c1]], [[arg0_val]]
// CHECK:     [[res:%[^ ]*]] = affine.for [[i:[^ ]*]] = 1 to 10 iter_args([[iter_arg:%[^ ]*]] = [[peeled]]) -> (i32) {
// CHECK:       arith.addi
// CHECK:       affine.yield
func.func @peel_first_iter(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c1 = arith.constant 1 : i32
  %0 = secret.generic(%arg0: !secret.secret<i32>) {
  ^body(%arg0_val: i32):
    %res = affine.for %i = 0 to 10 iter_args(%sum_iter = %c1) -> i32 {
      %sum = arith.addi %sum_iter, %arg0_val : i32
      affine.yield %sum : i32
    }
    secret.yield %res : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
