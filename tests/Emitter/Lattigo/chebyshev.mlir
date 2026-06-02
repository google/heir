// RUN: heir-translate %s --emit-lattigo | FileCheck %s

!evaluator = !lattigo.ckks.evaluator
!params = !lattigo.ckks.parameter
!eval = !lattigo.ckks.polynomial_evaluator
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.ckks} {
  // CHECK: func chebyshev_custom_domain
  // CHECK: [[out:ct[0-9]+]]_polyCoeffs := []*big.Float{
  // CHECK: [[out]]_interval := [2]float64{-2.000000e+00, 2.000000e+00}
  // CHECK: [[out]]_bignumPoly := bignum.NewPolynomial(bignum.Chebyshev, [[out]]_polyCoeffs, [[out]]_interval)
  // CHECK: [[out]], {{.*}} := {{.*}}.Evaluate(
  func.func @chebyshev_custom_domain(%params: !params, %evaluator: !evaluator, %ct: !ct) -> !ct {
    %eval = lattigo.ckks.new_polynomial_evaluator %params, %evaluator : (!params, !evaluator) -> !eval
    %0 = lattigo.ckks.chebyshev %eval, %ct {coefficients = [1.0, 0.5], targetScale = 1073741824, domain = array<f64: -2.0, 2.0>} : (!eval, !ct) -> !ct
    return %0 : !ct
  }

  // An unset domain passes nil, without emitting an interval.
  // CHECK: func chebyshev_unset_domain
  // CHECK-NOT: _interval :=
  // CHECK: [[out:ct[0-9]+]]_bignumPoly := bignum.NewPolynomial(bignum.Chebyshev, [[out]]_polyCoeffs, nil)
  func.func @chebyshev_unset_domain(%params: !params, %evaluator: !evaluator, %ct: !ct) -> !ct {
    %eval = lattigo.ckks.new_polynomial_evaluator %params, %evaluator : (!params, !evaluator) -> !eval
    %0 = lattigo.ckks.chebyshev %eval, %ct {coefficients = [1.0, 0.5], targetScale = 1073741824} : (!eval, !ct) -> !ct
    return %0 : !ct
  }
}
