// RUN: heir-opt --yosys-optimizer --canonicalize %s | FileCheck %s

// Tests Yosys Optimizer on single-bit secret inputs and outputs.

module {
  // CHECK: @bits
  func.func @bits(%in: !secret.secret<i1>) -> (!secret.secret<i1>) {
    %one = arith.constant 1 : i1
    // CHECK: [[V1:%.*]] = secret.generic
    %1 = secret.generic
        (%in: !secret.secret<i1>, %one: i1) {
        ^bb0(%IN: i1, %ONE: i1) :
            // CHECK-NOT: arith.addi
            %2 = arith.addi %IN, %ONE : i1
            secret.yield %2 : i1
        } -> (!secret.secret<i1>)
    // CHECK: return [[V1]]
    return %1 : !secret.secret<i1>
  }
}
