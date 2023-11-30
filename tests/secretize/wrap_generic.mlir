// RUN: heir-opt --split-input-file --wrap-generic %s | FileCheck %s

// CHECK: module
module {
    // CHECK: @main(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i1>) -> !secret.secret<i32>
    func.func @main(%value: i32 {secret.secret}, %cond: i1 {secret.secret}) -> (i32) {
      // CHECK: %[[V0:.*]] = secret.generic
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c7 = arith.constant 7 : i32
      %0 = arith.muli %value, %c7 : i32
      %1 = arith.addi %0, %c1 : i32
      %2 = arith.muli %1, %1 : i32
      %3 = arith.select %cond, %2, %c0 : i32
      // CHECK: return %[[V0]] : !secret.secret<i32>
      func.return %3 : i32
    }
}

// -----

module {
    // CHECK: @multiple_outputs(%arg0: !secret.secret<i32>) -> (!secret.secret<i1>, !secret.secret<i32>)
    func.func @multiple_outputs(%value: i32 {secret.secret}) -> (i1, i32) {
      // CHECK: %[[V0:.*]]:2 = secret.generic
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c7 = arith.constant 7 : i32
      %0 = arith.muli %value, %c7 : i32
      %1 = arith.addi %0, %c1 : i32
      %2 = arith.muli %1, %1 : i32
      %3 = arith.cmpi slt, %value, %c0 : i32
      // CHECK: return %[[V0]]#0, %[[V0]]#1 : !secret.secret<i1>, !secret.secret<i32>
      func.return %3, %2 : i1, i32
    }
}

// -----

module {
    // CHECK: @nonsecret_input(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: !secret.secret<i32>) -> !secret.secret<i32>
    func.func @nonsecret_input(%const: i32, %value: i32 {secret.secret}) -> i32 {
      // CHECK: %[[V0:.*]] = secret.generic ins(%[[ARG0]], %[[ARG1]] : i32, !secret.secret<i32>)
      %1 = arith.addi %const, %value : i32
      // CHECK: return %[[V0]] : !secret.secret<i32>
      func.return %1 : i32
    }
}
