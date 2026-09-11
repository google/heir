// RUN: heir-opt --split-input-file --polynomial-approximation=math-exp-method=taylor %s | FileCheck %s --check-prefix=CHECK-TAYLOR
// RUN: heir-opt --split-input-file --polynomial-approximation=math-exp-method=cf %s | FileCheck %s --check-prefix=CHECK-CF
// RUN: heir-opt --split-input-file --polynomial-approximation=math-exp-method=auto %s | FileCheck %s --check-prefix=CHECK-AUTO

// CHECK-TAYLOR: @test_exp_scalar
// CHECK-TAYLOR: %[[SCALE:.*]] = arith.constant 2.500000e-01 : f32
// CHECK-TAYLOR: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-TAYLOR: %[[SCALED:.*]] = arith.mulf %{{.*}}, %[[SCALE]] : f32
// CHECK-TAYLOR: %[[V0:.*]] = arith.addf %[[SCALED]], %[[ONE]] : f32
// CHECK-TAYLOR: %[[V1:.*]] = arith.mulf %[[V0]], %[[V0]] : f32
// CHECK-TAYLOR: %[[V2:.*]] = arith.mulf %[[V1]], %[[V1]] : f32
// CHECK-TAYLOR: return %[[V2]] : f32

// CHECK-CF: @test_exp_scalar
// CHECK-CF: %[[POLY:.*]] = polynomial.eval
// CHECK-CF-SAME: domain_lower = -1.000000e+00 : f64
// CHECK-CF-SAME: domain_upper = 1.000000e+00 : f64
// CHECK-CF-SAME: f32
// CHECK-CF: return %[[POLY]] : f32

// CHECK-AUTO: @test_exp_scalar
// CHECK-AUTO: %[[POLY:.*]] = polynomial.eval
// CHECK-AUTO-SAME: domain_lower = -1.000000e+00 : f64
// CHECK-AUTO-SAME: domain_upper = 1.000000e+00 : f64
// CHECK-AUTO-SAME: f32
// CHECK-AUTO: return %[[POLY]] : f32
func.func @test_exp_scalar(%x: f32 {secret.secret}) -> f32 {
  %0 = math.exp %x {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f32
  return %0 : f32
}

// -----

// CHECK-TAYLOR: @test_exp_tensor
// CHECK-TAYLOR: %[[SCALE:.*]] = arith.constant dense<7.812500e-03> : tensor<4xf32>
// CHECK-TAYLOR: %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-TAYLOR: %[[SCALED:.*]] = arith.mulf %{{.*}}, %[[SCALE]] : tensor<4xf32>
// CHECK-TAYLOR: %[[V0:.*]] = arith.addf %[[SCALED]], %[[ONE]] : tensor<4xf32>
// CHECK-TAYLOR: %[[V1:.*]] = arith.mulf %[[V0]], %[[V0]] : tensor<4xf32>
// CHECK-TAYLOR: return

// CHECK-CF: @test_exp_tensor
// CHECK-CF: %[[POLY:.*]] = polynomial.eval
// CHECK-CF-SAME: tensor<4xf32>
// CHECK-CF: return %[[POLY]] : tensor<4xf32>

// CHECK-AUTO: @test_exp_tensor
// CHECK-AUTO: %[[POLY:.*]] = polynomial.eval
// CHECK-AUTO-SAME: tensor<4xf32>
// CHECK-AUTO: return %[[POLY]] : tensor<4xf32>
func.func @test_exp_tensor(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  %0 = math.exp %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-TAYLOR: @test_exp_taylor_fallback_out_of_bounds
// CHECK-TAYLOR: polynomial.eval
// CHECK-TAYLOR-SAME: domain_upper = 2
func.func @test_exp_taylor_fallback_out_of_bounds(%x: f32 {secret.secret}) -> f32 {
  %0 = math.exp %x {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// CHECK-TAYLOR: @test_exp_pinned_chebyshev
// CHECK-TAYLOR: %[[POLY:.*]] = polynomial.eval
// CHECK-TAYLOR-SAME: f32
// CHECK-TAYLOR: return %[[POLY]] : f32
func.func @test_exp_pinned_chebyshev(%x: f32 {secret.secret}) -> f32 {
  %0 = math.exp %x {approximation_method = "chebyshev", degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f32
  return %0 : f32
}
