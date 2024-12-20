// RUN: heir-opt --arith-quarter-wide-int  %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: func @test_simple_split
func.func @test_simple_split(%arg0: i32, %arg1: i32) -> i32 {
  %1 = arith.constant 522067228: i32 // Hex 1f1e1d1c
  %2 = arith.constant 31 : i8
  %3 = arith.extsi %2 : i8 to i32
  %4 = arith.muli %1, %arg1 : i32
  %5 = arith.addi %arg0, %3 : i32
  return %4 : i32
}
