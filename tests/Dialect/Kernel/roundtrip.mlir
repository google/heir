// RUN: heir-opt %s | FileCheck %s

// CHECK: module
module {
  // CHECK: @test_chebyshev
  func.func @test_chebyshev(%arg0: f64) -> f64 {
    // CHECK: kernel.eval_chebyshev %arg0 {coefficients = [1.000000e+00, 2.000000e+00]} : f64 -> f64
    %0 = kernel.eval_chebyshev %arg0 {coefficients = [1.0, 2.0]} : f64 -> f64
    return %0 : f64
  }
}
