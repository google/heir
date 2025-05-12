// RUN: heir-opt --canonicalize %s | FileCheck %s

// CHECK: func @remove_unused_yielded_values
func.func @remove_unused_yielded_values(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %X = arith.constant 7 : i32
  %Y = secret.conceal %X : i32 -> !secret.secret<i32>
  %Z, %UNUSED = secret.generic
    (%Y: !secret.secret<i32>, %arg0: !secret.secret<i32>) {
    ^bb0(%y: i32, %clear_arg0 : i32) :
      %d = arith.addi %clear_arg0, %y: i32
      %unused = arith.addi %y, %y: i32
      // CHECK: secret.yield %[[value:.*]] : i32
      secret.yield %d, %unused : i32, i32
    } -> (!secret.secret<i32>, !secret.secret<i32>)
  return %Z : !secret.secret<i32>
}

// CHECK: func @remove_pass_through_args
func.func @remove_pass_through_args(
// CHECK: %[[arg1:.*]]: !secret.secret<i32>, %[[arg2:.*]]: !secret.secret<i32>
    %arg1 : !secret.secret<i32>, %arg2 : !secret.secret<i32>) -> (!secret.secret<i32>, !secret.secret<i32>) {
  // CHECK: %[[out1:.*]] = secret.generic
  %out1, %out2 = secret.generic
    (%arg1: !secret.secret<i32>, %arg2: !secret.secret<i32>) {
    ^bb0(%x: i32, %y: i32) :
      // CHECK: %[[value:.*]] = arith.addi
      %z = arith.addi %x, %y : i32
      // Only yield one value
      // CHECK: secret.yield %[[value]] : i32
      secret.yield %z, %y : i32, i32
    } -> (!secret.secret<i32>, !secret.secret<i32>)
  // CHECK: return %[[out1]], %[[arg2]]
  return %out1, %out2 : !secret.secret<i32>, !secret.secret<i32>
}
