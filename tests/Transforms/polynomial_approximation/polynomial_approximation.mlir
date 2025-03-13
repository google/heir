// RUN: heir-opt --polynomial-approximation %s | FileCheck %s

// CHECK-LABEL: @test_exp
func.func @test_exp(%x: f32) -> f32 {
  // Don't assert the quality of the approximation, just that it was applied
  // and has the right degree. Leave quality-of-approximation for unit testing.
  // CHECK: polynomial.eval
  // CHECK-SAME: x**3
  // CHECK-NOT: x**4
  %0 = math.exp %x {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f32
  return %0 : f32
}

// CHECK-LABEL: @test_sin_default_params
func.func @test_sin_default_params(%x: f32) -> f32 {
  // CHECK: polynomial.eval
  // CHECK-SAME: x**5
  // CHECK-NOT: x**6
  %0 = math.sin %x : f32
  return %0 : f32
}
