// RUN: heir-opt -yosys-optimizer %s | FileCheck %s

  // CHECK-LABEL: @for
  func.func @for_loop(%arg0: i8, %arg1: i8) -> i32 {
    // CHECK-NOT: arith.extsi
    // CHECK-NOT: arith.subi
    // CHECK-NOT: arith.muli
    // CHECK-NOT: arith.addi
    %c-128_i16 = arith.constant -128 : i16
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.extsi %arg0 : i8 to i16
    %1 = arith.subi %0, %c-128_i16 : i16
    %2 = arith.extsi %1 : i16 to i32
    %3 = arith.extsi %arg1 : i8 to i32
    %4 = arith.muli %2, %3 : i32
    %5 = arith.addi %c0_i32, %4 : i32
    // CHECK: return
    return %5 : i32
  }
