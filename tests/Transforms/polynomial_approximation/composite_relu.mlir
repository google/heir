// RUN: heir-opt --split-input-file --polynomial-approximation=use-composite-relu=true %s | FileCheck %s

// Composite-sign ReLU on a calibrated (non-[-1,1]) domain. The 1/B prescale is
// materialized explicitly (B = max|domain| = 3.527138, so 1/B = 0.2835160) and
// each Chebyshev sign stage is emitted on [-1, 1] so the kept preserve-poly-eval /
// cheddar eval_poly (which only evaluates on [-1, 1]) can consume it. The final
// ReLU multiply uses the original (un-prescaled) input.
// CHECK: @test_composite_relu_prescale
func.func @test_composite_relu_prescale(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  // CHECK-DAG: %[[INVB:.*]] = arith.constant dense<2.835160e-01>
  // CHECK: %[[PS:.*]] = arith.mulf %[[ARG:.*]], %[[INVB]]
  // CHECK: %[[S0:.*]] = polynomial.eval
  // CHECK-SAME: %[[PS]]
  // CHECK-SAME: domain_lower = -1.000000e+00
  // CHECK-SAME: domain_upper = 1.000000e+00
  // CHECK: %[[S1:.*]] = polynomial.eval {{.*}}%[[S0]]
  // CHECK-SAME: domain_lower = -1.000000e+00
  // CHECK: %[[STEP:.*]] = polynomial.eval {{.*}}%[[S1]]
  // CHECK-SAME: domain_lower = -1.000000e+00
  // CHECK: arith.mulf %[[ARG]], %[[STEP]]
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %r = arith.maximumf %x, %c0 {domain_lower = -3.527138 : f64, domain_upper = 3.527138 : f64} : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Symmetric default domain [-1, 1]: prescale is a no-op (1/B = 1.0), so no
// arith.mulf prescale is emitted and the first sign stage consumes the input
// directly.
// CHECK: @test_composite_relu_unit_domain
func.func @test_composite_relu_unit_domain(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  // CHECK-NOT: arith.mulf %arg0, %{{.*}}cst
  // CHECK: polynomial.eval
  // CHECK-SAME: domain_lower = -1.000000e+00
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %r = arith.maximumf %x, %c0 {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : tensor<4xf32>
  return %r : tensor<4xf32>
}
