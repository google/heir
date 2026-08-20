// RUN: heir-translate --emit-tfhe-rust-hl %s | FileCheck %s

// CHECK: pub fn f(
// CHECK: for [[I:.*]] in 0..16 {
// CHECK-NEXT: }
func.func @f() {
  affine.for %i = 0 to 16 {
  }
  return
}
