// RUN: heir-opt --lower-polynomial-eval="method=pscheb" %s | FileCheck %s

!poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>
#poly = #polynomial.typed_chebyshev_polynomial<[0.0, 0.75, 0.0, 0.25]> : !poly_ty

module {
  func.func @chebyshev(%ct: f32) -> f32 {
    %ct_0 = polynomial.eval #poly, %ct {coefficients = [0.0, 0.75, 0.0, 0.25], domain_lower = -1.000000e+00 : f64, domain_upper = 1.000000e+00 : f64} : f32
    return %ct_0 : f32
  }
}