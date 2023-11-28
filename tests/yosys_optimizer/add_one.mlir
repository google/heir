// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

// CHECK: module
module {
  func.func @add_one(%in: i8) -> (i8) {
    // CHECK: comb.truth_table
    %0 = arith.constant 1 : i8
    // CHECK-NOT arith.addi
    %1 = arith.addi %in, %0 : i8
    // CHECK: tensor.from_elements
    // CHECK-NEXT: return
    return %1 : i8
  }
}
