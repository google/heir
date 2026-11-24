// RUN: heir-opt --split-input-file --verify-diagnostics --polynomial-approximation %s | FileCheck %s

// levels = 5: a degree-3 Chebyshev initial guess (2 levels), 1 level to set up
// the error term, and 2 Goldschmidt iterations. The iterations are unrolled,
// since layout propagation does not support loops, and the last one skips the
// unused e * e.
// CHECK: @test_inverse_scalar
// CHECK-SAME: (%[[X:.*]]: f32
func.func @test_inverse_scalar(%x: f32 {secret.secret}) -> f32 {
  // CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[Y0:.*]] = polynomial.eval
  // CHECK-SAME: typed_chebyshev_polynomial <[{{.*}}, {{.*}}, {{.*}}, {{.*}}]>
  // CHECK-SAME: %[[X]]
  // CHECK-SAME: domain_lower = 5.000000e-01 : f64
  // CHECK-SAME: domain_upper = 2.000000e+00 : f64
  // CHECK: %[[XY0:.*]] = arith.mulf %[[X]], %[[Y0]] : f32
  // CHECK: %[[E0:.*]] = arith.subf %[[ONE]], %[[XY0]] : f32
  // CHECK: %[[P0:.*]] = arith.addf %[[E0]], %[[ONE]] : f32
  // CHECK: %[[Y1:.*]] = arith.mulf %[[Y0]], %[[P0]] : f32
  // CHECK: %[[E1:.*]] = arith.mulf %[[E0]], %[[E0]] : f32
  // CHECK: %[[P1:.*]] = arith.addf %[[E1]], %[[ONE]] : f32
  // CHECK: %[[RESULT:.*]] = arith.mulf %[[Y1]], %[[P1]] : f32
  // CHECK-NOT: scf.for
  // CHECK-NOT: arith.divf
  // CHECK: return %[[RESULT]] : f32
  %one = arith.constant 1.0 : f32
  %0 = arith.divf %one, %x {levels = 5 : i64, domain_lower = 0.5 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// levels = 3: a degree-1 guess and a single Goldschmidt iteration, which is
// emitted directly without a loop.
// CHECK: @test_inverse_single_iteration
// CHECK-SAME: (%[[X:.*]]: f32
func.func @test_inverse_single_iteration(%x: f32 {secret.secret}) -> f32 {
  // CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[Y0:.*]] = polynomial.eval
  // CHECK-SAME: typed_chebyshev_polynomial <[{{.*}}, {{.*}}]>
  // CHECK: %[[XY0:.*]] = arith.mulf %[[X]], %[[Y0]] : f32
  // CHECK: %[[E0:.*]] = arith.subf %[[ONE]], %[[XY0]] : f32
  // CHECK-NOT: scf.for
  // CHECK: %[[P:.*]] = arith.addf %[[E0]], %[[ONE]] : f32
  // CHECK: %[[RESULT:.*]] = arith.mulf %[[Y0]], %[[P]] : f32
  // CHECK: return %[[RESULT]] : f32
  %one = arith.constant 1.0 : f32
  %0 = arith.divf %one, %x {levels = 3 : i64, domain_lower = 0.5 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// Tensor input with no attributes: default levels = 7 (a degree-7 guess and 3
// unrolled iterations) and the default positive domain [0.1, 2.0].
// CHECK: @test_inverse_tensor
func.func @test_inverse_tensor(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  // CHECK: arith.constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK: polynomial.eval
  // CHECK-SAME: typed_chebyshev_polynomial <[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]>
  // CHECK-SAME: domain_lower = 1.000000e-01 : f64
  // CHECK-SAME: domain_upper = 2.000000e+00 : f64
  // CHECK-SAME: tensor<4xf32>
  // CHECK-COUNT-3: arith.addf {{.*}} : tensor<4xf32>
  // CHECK-NOT: scf.for
  // CHECK-NOT: arith.divf
  // CHECK: return
  %one = arith.constant dense<1.0> : tensor<4xf32>
  %0 = arith.divf %one, %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A numerator other than 1 is decomposed as 2 * (1 / x), and the reciprocal is
// approximated. The domain attributes describe the divisor, so they move to the
// reciprocal.
// CHECK: @test_numerator_not_one
// CHECK-SAME: (%[[X:.*]]: f32
func.func @test_numerator_not_one(%x: f32 {secret.secret}) -> f32 {
  // CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
  // CHECK: polynomial.eval
  // CHECK-SAME: %[[X]]
  // CHECK-SAME: domain_lower = 5.000000e-01 : f64
  // CHECK-SAME: domain_upper = 2.000000e+00 : f64
  // CHECK: %[[RESULT:.*]] = arith.mulf %{{.*}}, %[[TWO]] : f32
  // CHECK-NOT: arith.divf
  // CHECK: return %[[RESULT]] : f32
  %two = arith.constant 2.0 : f32
  %0 = arith.divf %two, %x {domain_lower = 0.5 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// A plaintext denominator does not need approximating.
// CHECK: @test_denominator_not_secret
func.func @test_denominator_not_secret(%x: f32) -> f32 {
  // CHECK-NOT: polynomial.eval
  // CHECK: arith.divf
  %one = arith.constant 1.0 : f32
  %0 = arith.divf %one, %x {domain_lower = 0.5 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// Fewer than 3 levels leaves no room for a Goldschmidt iteration.
func.func @too_few_levels(%x: f32 {secret.secret}) -> f32 {
  %one = arith.constant 1.0 : f32
  // expected-error@+1 {{Must allocate at least 3 levels}}
  %0 = arith.divf %one, %x {levels = 2 : i64, domain_lower = 0.5 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// 1/x is undefined at 0, so the domain must be strictly positive.
func.func @zero_lower_bound(%x: f32 {secret.secret}) -> f32 {
  %one = arith.constant 1.0 : f32
  // expected-error@+1 {{domain_lower must be strictly greater than 0}}
  %0 = arith.divf %one, %x {domain_lower = 0.0 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// A domain containing the pole at 0 is rejected.
func.func @domain_contains_zero(%x: f32 {secret.secret}) -> f32 {
  %one = arith.constant 1.0 : f32
  // expected-error@+1 {{domain_lower must be strictly greater than 0}}
  %0 = arith.divf %one, %x {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f32
  return %0 : f32
}

// -----

// A division of two secrets (e.g., torch's x / y) is decomposed as
// x * (1 / y), and the reciprocal is approximated.
// CHECK: @test_secret_numerator
// CHECK-SAME: (%[[X:.*]]: tensor<4xf32> {secret.secret}, %[[Y:.*]]: tensor<4xf32>
func.func @test_secret_numerator(%x: tensor<4xf32> {secret.secret}, %y: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  // CHECK: polynomial.eval
  // CHECK-SAME: %[[Y]]
  // CHECK: %[[RESULT:.*]] = arith.mulf %[[X]], %{{.*}} : tensor<4xf32>
  // CHECK-NOT: arith.divf
  // CHECK: return %[[RESULT]] : tensor<4xf32>
  %0 = arith.divf %x, %y : tensor<4xf32>
  return %0 : tensor<4xf32>
}
