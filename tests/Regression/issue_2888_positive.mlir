// Regression test for issue #2888 (positive case).

// RUN: heir-opt %s --polynomial-approximation --lower-polynomial-eval | FileCheck %s

func.func @sqrt_on_positive_domain(%x: f32 {secret.secret}) -> f32 {
  // CHECK-NOT: math.sqrt
  %0 = math.sqrt %x {domain_lower = 0.25 : f64, domain_upper = 4.0 : f64} : f32
  return %0 : f32
}
