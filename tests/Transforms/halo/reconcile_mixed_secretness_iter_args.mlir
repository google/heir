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


// CHECK: @dont_peel_secret([[arg0:%[^:]*]]: !secret.secret<i32>, [[init:%[^ ]*]]: !secret.secret<i32>
// CHECK:   [[res:%[^ ]*]] = secret.generic([[arg0]]
// CHECK:   ^[[body:[^(]*]]([[arg0_val:%[^:]*]]: i32, [[init_val:%[^:]*]]: i32):
// CHECK-NEXT:     affine.for [[i:[^ ]*]] = 0 to 10 iter_args([[iter_arg:%[^ ]*]] = [[init_val]]) -> (i32) {
// CHECK:       arith.addi
// CHECK:       affine.yield
func.func @dont_peel_secret(%arg0: !secret.secret<i32>, %init: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0: !secret.secret<i32>, %init: !secret.secret<i32>) {
  ^body(%arg0_val: i32, %init_val: i32):
    %res = affine.for %i = 0 to 10 iter_args(%sum_iter = %init_val) -> i32 {
      %sum = arith.addi %sum_iter, %arg0_val : i32
      affine.yield %sum : i32
    }
    secret.yield %res : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}


// CHECK: @dont_peel_nonsecret([[arg0:%[^:]*]]: i32
// CHECK-NEXT:  arith.constant
// CHECK-NEXT:  affine.for [[i:[^ ]*]] = 0 to 10
func.func @dont_peel_nonsecret(%arg0: i32) -> i32 {
  %init = arith.constant 1 : i32
  %0 = affine.for %i = 0 to 10 iter_args(%sum_iter = %init) -> i32 {
    %sum = arith.addi %sum_iter, %arg0 : i32
    affine.yield %sum : i32
  }
  return %0 : i32
}

// CHECK: @peel_first_iter_scf([[arg0:%[^:]*]]: !secret.secret<i32>)
// CHECK:   [[c0:%[^ ]*]] = arith.constant 0 : index
// CHECK:   [[c1:%[^ ]*]] = arith.constant 1 : index
// CHECK:   [[c1_i32:%[^ ]*]] = arith.constant 1 : i32
// CHECK:   [[c10:%[^ ]*]] = arith.constant 10 : index
// CHECK:   [[res:%[^ ]*]] = secret.generic([[arg0]]
// CHECK:   ^[[body:[^(]*]]([[arg0_val:%[^:]*]]: i32):
// CHECK:     [[c1_again:%[^ ]*]] = arith.constant 1 : index
// CHECK:     [[peeled:%[^ ]*]] = arith.addi [[c1_i32]], [[arg0_val]]
// CHECK:     [[res:%[^ ]*]] = scf.for [[i:[^ ]*]] = [[c1_again]] to [[c10]] step [[c1]] iter_args([[iter_arg:%[^ ]*]] = [[peeled]]) -> (i32) {
// CHECK:       arith.addi
// CHECK:       scf.yield
func.func @peel_first_iter_scf(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %0 = secret.generic(%arg0: !secret.secret<i32>) {
  ^body(%arg0_val: i32):
    %res = scf.for %i = %c0 to %c10 step %c1 iter_args(%sum_iter = %c1_i32) -> i32 {
      %sum = arith.addi %sum_iter, %arg0_val : i32
      scf.yield %sum : i32
    }
    secret.yield %res : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}


// CHECK: @dont_peel_secret_scf([[arg0:%[^:]*]]: !secret.secret<i32>, [[init:%[^ ]*]]: !secret.secret<i32>
// CHECK:   [[c0:%[^ ]*]] = arith.constant 0 : index
// CHECK:   [[c1:%[^ ]*]] = arith.constant 1 : index
// CHECK:   [[c10:%[^ ]*]] = arith.constant 10 : index
// CHECK:   [[res:%[^ ]*]] = secret.generic([[arg0]]
// CHECK:   ^[[body:[^(]*]]([[arg0_val:%[^:]*]]: i32, [[init_val:%[^:]*]]: i32):
// CHECK-NEXT:     scf.for [[i:[^ ]*]] = [[c0]] to [[c10]] step [[c1]] iter_args([[iter_arg:%[^ ]*]] = [[init_val]]) -> (i32) {
// CHECK:       arith.addi
// CHECK:       scf.yield
func.func @dont_peel_secret_scf(%arg0: !secret.secret<i32>, %init: !secret.secret<i32>) -> !secret.secret<i32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = secret.generic(%arg0: !secret.secret<i32>, %init: !secret.secret<i32>) {
  ^body(%arg0_val: i32, %init_val: i32):
    %res = scf.for %i = %c0 to %c10 step %c1 iter_args(%sum_iter = %init_val) -> i32 {
      %sum = arith.addi %sum_iter, %arg0_val : i32
      scf.yield %sum : i32
    }
    secret.yield %res : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}


// CHECK: @dont_peel_nonsecret_scf([[arg0:%[^:]*]]: i32
// CHECK:   [[c0:%[^ ]*]] = arith.constant 0 : index
// CHECK:   [[c1:%[^ ]*]] = arith.constant 1 : index
// CHECK:   [[c10:%[^ ]*]] = arith.constant 10 : index
// CHECK:   [[init:%[^ ]*]] = arith.constant 1 : i32
// CHECK:   scf.for [[i:[^ ]*]] = [[c0]] to [[c10]] step [[c1]] iter_args(%{{.*}} = [[init]])
func.func @dont_peel_nonsecret_scf(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 1 : i32
  %0 = scf.for %i = %c0 to %c10 step %c1 iter_args(%sum_iter = %init) -> i32 {
    %sum = arith.addi %sum_iter, %arg0 : i32
    scf.yield %sum : i32
  }
  return %0 : i32
}
