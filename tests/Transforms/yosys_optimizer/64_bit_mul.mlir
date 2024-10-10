// RUN: heir-opt --yosys-optimizer --canonicalize --cse %s | FileCheck %s

module {
  // CHECK-LABEL: @mul
  func.func @mul(%x: !secret.secret<i64>, %y: !secret.secret<i64>) -> (!secret.secret<i64>) {
    // Generic to convert the i64 to a memref
    // CHECK: secret.cast
    // CHECK-SAME: !secret.secret<i64> to !secret.secret<memref<64xi1>>

    // CHECK: secret.generic
    %1 = secret.generic
        ins(%x, %y: !secret.secret<i64>, !secret.secret<i64>) {
        ^bb0(%X: i64, %Y: i64) :
            // CHECK-NOT: arith.addi

            // CHECK-COUNT-6464: comb.truth_table
            // CHECK-NOT: comb.truth_table
            %0 = arith.extsi %X : i64 to i128
            %1 = arith.extsi %Y : i64 to i128
            %2 = arith.muli %0, %1 : i128
            %3 = arith.trunci %2 : i128 to i64
            secret.yield %3 : i64
        } -> (!secret.secret<i64>)

    // CHECK: secret.cast
    // CHECK-SAME: !secret.secret<memref<64xi1>> to !secret.secret<i64>
    return %1 : !secret.secret<i64>
  }
}
