// RUN: heir-opt --lower-polynomial-eval="method=pscheb" %s | FileCheck %s

!poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>
// This Chebyshev polynomial [0.0, 0.75, 0.0, 0.25] is equivalent to x^3 in monomial basis
#poly = #polynomial.typed_chebyshev_polynomial<[0.0, 0.75, 0.0, 0.25]> : !poly_ty

module {
  // CHECK-LABEL: @chebyshev
  func.func @chebyshev(%arg0: f32) -> f32 {
    // When the monomial basis is simpler (like x^3), we should use it instead of PS-Cheb
    // CHECK-NOT: arith.constant 5.000000e-01
    // CHECK-NOT: arith.constant 2.000000e+00
    // CHECK: [[V0:%.+]] = arith.mulf %arg0, %arg0
    // CHECK-NEXT: [[V1:%.+]] = arith.mulf [[V0]], %arg0
    // CHECK-NEXT: return [[V1]]
    %ct_0 = polynomial.eval #poly, %arg0 {coefficients = [0.0, 0.75, 0.0, 0.25], domain_lower = -1.000000e+00 : f64, domain_upper = 1.000000e+00 : f64} : f32
    return %ct_0 : f32
  }
}
