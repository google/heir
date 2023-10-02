// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

// CHECK: module
module {
    func.func @ops(i32, i32, i32, i32) -> (i32) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32):
    // CHECK: comb.truth_table
    %0 = arith.subi %arg0, %arg1: i32
    %1 = arith.muli %arg2, %arg3 : i32
    %2 = arith.andi %1, %arg3 : i32
    // CHECK: return
    return %2 : i32
    }
}
