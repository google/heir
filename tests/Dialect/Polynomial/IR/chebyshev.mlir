// RUN: heir-opt %s | FileCheck %s

// This file tests polynomial syntax for chebyshev polynomial attributes

!polyty = !polynomial.polynomial<ring=<coefficientType=f64>>
#chebpoly = #polynomial.typed_chebyshev_polynomial<[1.0, 2.0, 3.0, 4.0]> : !polyty

// CHECK: test_eval
func.func @test_eval(%arg0: f64) -> f64 {
  %0 = polynomial.eval #chebpoly, %arg0 : f64
  return %0 : f64
}
