
// RUN: heir-opt --arith-quarter-wide-int -convert-tensor-to-scalars -convert-to-llvm %s | mlir-translate --mlir-to-llvmir | llc -filetype=obj > %t
// RUN: clang -c quarter_to_llvm.c
// RUN: clang quarter_to_llvm.o %t -o a.out
// RUN: ./a.out | FileCheck %s

// 1f1e1d1c * fb + 9f
// 1e82868a74 + 9f
// CHECK: 82 86 8b 13
func.func @test_lowera_quarter_mul(%arg0: i32) -> i32{
  %c1 = arith.constant 522067228: i32 // Hex 1f1e1d1c
  %c2 = arith.constant 159 : i8
  %3 = arith.extui %c2 : i8 to i32
  %4 = arith.muli %c1, %arg0 : i32
  %5 = arith.addi %4, %3 : i32
  return %5 : i32
}
