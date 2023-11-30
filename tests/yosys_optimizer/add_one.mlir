// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

module {
  // CHECK-LABEL: @add_one
  func.func @add_one(%in: !secret.secret<i8>) -> (!secret.secret<i8>) {
    %one = arith.constant 1 : i8
    // Generic to convert the i8 to a tensor
    // CHECK: secret.cast
    // CHECK-SAME: !secret.secret<i8> to !secret.secret<tensor<8xi1>>

    // CHECK: secret.generic
    %1 = secret.generic
        ins(%in, %one: !secret.secret<i8>, i8) {
        ^bb0(%IN: i8, %ONE: i8) :
            // CHECK-NOT: arith.addi
            %2 = arith.addi %IN, %ONE : i8
            secret.yield %2 : i8
        } -> (!secret.secret<i8>)

    // CHECK: secret.cast
    // CHECK-SAME: !secret.secret<tensor<8xi1>> to !secret.secret<i8>
    return %1 : !secret.secret<i8>
  }
}
