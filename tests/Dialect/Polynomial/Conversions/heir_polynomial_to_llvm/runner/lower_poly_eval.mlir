!poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>

module {
  func.func @test_poly_eval(%arg0: f32) -> f32  {
    %result = polynomial.eval #polynomial.typed_float_polynomial<1.0 + 1.0 x + 1.0 x**2> : !poly_ty, %arg0 : f32
    return %result : f32
  }
}
