// RUN: heir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @remove_unused_yielded_values
func.func @remove_unused_yielded_values(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %X = arith.constant 7 : i32
  %Y = secret.conceal %X : i32 -> !secret.secret<i32>
  %Z, %UNUSED = secret.generic
    ins(%Y, %arg0 : !secret.secret<i32>, !secret.secret<i32>) {
    ^bb0(%y: i32, %clear_arg0 : i32) :
      %d = arith.addi %clear_arg0, %y: i32
      %unused = arith.addi %y, %y: i32
      // CHECK: secret.yield %[[value:.*]] : i32
      secret.yield %d, %unused : i32, i32
    } -> (!secret.secret<i32>, !secret.secret<i32>)
  func.return %Z : !secret.secret<i32>
}
