// RUN: heir-opt --polynomial-approximation %s | FileCheck %s

// CHECK: @test_exp
func.func @test_exp(%x: f32) -> f32 {
  // CHECK: polynomial.eval
  // CHECK-SAME: x**3
  // CHECK-NOT: x**4
  %0 = math.exp %x {
      degree = 3 : i32,
      domain_lower = -1.0 : f64,
      domain_upper = 1.0 : f64} : f32
  return %0 : f32
}
