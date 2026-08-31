// RUN: heir-translate --emit-tfhe-rust-hl %s | FileCheck %s

// CHECK: pub fn multiple_iter_args(
// CHECK: let [[INIT0:.*]] = 0u32;
// CHECK-NEXT: let [[INIT1:.*]] = 1u32;
// CHECK-NEXT: let ([[RESULT0:.*]], [[RESULT1:.*]]) = (0..16).fold(([[INIT0]], [[INIT1]]), |(mut [[ITER0:.*]], mut [[ITER1:.*]]), [[I:.*]]| {
// CHECK-NEXT: ([[ITER0]], [[ITER1]])
// CHECK-NEXT: });
// CHECK-NEXT: ([[RESULT0]], [[RESULT1]])
func.func @multiple_iter_args() -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %result0, %result1 = affine.for %i = 0 to 16 iter_args(%iter0 = %c0, %iter1 = %c1) -> (i32, i32) {
    affine.yield %iter0, %iter1 : i32, i32
  }
  return %result0, %result1 : i32, i32
}
