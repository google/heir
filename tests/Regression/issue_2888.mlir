// Regression test for issue #2888.
//
// If polynomial-approximation is asked to approximate a function outside its
// domain of definition (e.g. sqrt on [-1, 1]), the Remez/CF routine samples
// the function where it is undefined and all Chebyshev coefficients come out
// as NaN. Lowering these NaN coefficients via Paterson-Stockmeyer used to
// trip `assert(node != nullptr)` in CachingVisitor::process.

// RUN: heir-opt %s --polynomial-approximation --lower-polynomial-eval --verify-diagnostics --split-input-file

!poly_ty = !polynomial.polynomial<ring=<coefficientType=f64>>
func.func @monomial_nan_direct(%x: f64) -> f64 {
  // expected-error@+1 {{non-finite}}
  %0 = polynomial.eval #polynomial.typed_float_polynomial<
      0x7FF8000000000000
      + 0x7FF8000000000000 x
      + 0x7FF8000000000000 x**2
  > : !poly_ty, %x : f64
  return %0 : f64
}

// -----

!poly_ty = !polynomial.polynomial<ring=<coefficientType=f64>>
func.func @chebyshev_nan_direct(%x: f64) -> f64 {
  // expected-error@+1 {{non-finite}}
  %0 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[
      0x7FF8000000000000 : f64, 0x7FF8000000000000 : f64,
      0x7FF8000000000000 : f64, 0x7FF8000000000000 : f64,
      0x7FF8000000000000 : f64, 0x7FF8000000000000 : f64
  ]> : !poly_ty, %x {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f64
  return %0 : f64
}

// -----
func.func @sqrt_on_negative_domain(%x: f32) -> f32 {
  // expected-error@+1 {{non-finite}}
  %0 = math.sqrt %x {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f32
  return %0 : f32
}

// -----
func.func @sqrt_on_positive_domain(%x: f32) -> f32 {
  // CHECK-NOT: math.sqrt
  %0 = math.sqrt %x {domain_lower = 0.25 : f64, domain_upper = 4.0 : f64} : f32
  return %0 : f32
}
