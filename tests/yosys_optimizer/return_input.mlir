// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

// Because these are trivial functions, Yosys does not change the program. This
// test is simply to ensure that the pass infrastructure handles these cases
// correctly.

// CHECK: module
module {
  // CHECK: func.func @return_wire
  func.func @return_wire(%arg0: i8) -> (i8) {
    // CHECK-NEXT: return %arg0
    return %arg0 : i8
  }
  // CHECK: func.func @return_constant_bit
  func.func @return_constant_bit() -> (i1) {
    %0 = arith.constant 0 : i1
    // CHECK: return
    return %0 : i1
  }
  // CHECK: func.func @return_bit
  func.func @return_bit(%arg0 : i1) -> (i1) {
    // CHECK-NEXT: return %arg0
    return %arg0 : i1
  }
}
