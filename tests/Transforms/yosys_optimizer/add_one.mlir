// RUN: heir-opt --yosys-optimizer --canonicalize --cse %s | FileCheck %s --check-prefix=CHECK --check-prefix=LUT
// RUN: heir-opt --yosys-optimizer="abc-fast=True" --canonicalize --cse %s | FileCheck %s --check-prefix=CHECK --check-prefix=LUT-FAST
// RUN: heir-opt --yosys-optimizer="mode=Boolean" --canonicalize --cse %s | FileCheck --check-prefix=CHECK --check-prefix=BOOL %s
// RUN: heir-opt --yosys-optimizer="mode=Boolean abc-fast=True" --canonicalize --cse %s | FileCheck --check-prefix=CHECK --check-prefix=BOOL-FAST %s

module {
  // CHECK: @add_one
  func.func @add_one(%in: !secret.secret<i8>) -> (!secret.secret<i8>) {
    %one = arith.constant 1 : i8
    // Generic to convert the i8 to a tensor
    // CHECK: secret.cast
    // CHECK-SAME: !secret.secret<i8> to !secret.secret<tensor<8xi1>>

    // CHECK: secret.generic
    %1 = secret.generic
        (%in: !secret.secret<i8>, %one: i8) {
        ^bb0(%IN: i8, %ONE: i8) :
            // CHECK-NOT: arith.addi

            // LUT-COUNT-11: comb.truth_table
            // LUT-FAST-COUNT-13: comb.truth_table
            // BOOL-COUNT-14: comb
            // BOOL-FAST-COUNT-16: comb

            // LUT-NOT: comb.truth_table
            // LUT-FAST-NOT: comb.truth_table
            // BOOL-NOT: comb
            // BOOL-FAST-NOT: comb
            %2 = arith.addi %IN, %ONE : i8
            secret.yield %2 : i8
        } -> (!secret.secret<i8>)

    // CHECK: secret.cast
    // CHECK-SAME: !secret.secret<tensor<8xi1>> to !secret.secret<i8>
    return %1 : !secret.secret<i8>
  }
}
