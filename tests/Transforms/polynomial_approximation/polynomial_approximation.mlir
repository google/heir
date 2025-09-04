// RUN: heir-opt --split-input-file --polynomial-approximation %s | FileCheck %s

// CHECK: @test_exp
func.func @test_exp(%x: f32) -> f32 {
  // Don't assert the quality of the approximation, just that it was applied
  // and has the right degree. Leave quality-of-approximation for unit testing.
  // CHECK: polynomial.eval
  // CHECK-SAME: [{{.*}}, {{.*}}, {{.*}}, {{.*}}]
  %0 = math.exp %x {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f32
  return %0 : f32
}

// -----

// CHECK: @test_domain
func.func @test_domain(%x: f32) -> f32 {
  // CHECK: polynomial.eval
  // CHECK-SAME: domain_upper = 2
  %0 = math.exp %x {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// CHECK: @test_sin_default_params
func.func @test_sin_default_params(%x: f32) -> f32 {
  // CHECK: polynomial.eval
  // CHECK-SAME: [{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]
  %0 = math.sin %x : f32
  return %0 : f32
}

// -----

// CHECK: @test_maximumf
func.func @test_maximumf(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: polynomial.eval
  // CHECK-NOT: arith.maximumf
  %c0 = arith.constant dense<0.0> : tensor<10xf32>
  %0 = arith.maximumf %x, %c0 : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// CHECK: @test_maximumf_domain
func.func @test_maximumf_domain(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: polynomial.eval
  // CHECK-SAME: domain_upper = 2
  // CHECK-NOT: arith.maximumf
  %c0 = arith.constant dense<0.0> : tensor<10xf32>
  %0 = arith.maximumf %x, %c0 {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 2.0 : f64}: tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----


// CHECK: @test_maximumf_ignore_not_splat
func.func @test_maximumf_ignore_not_splat(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: polynomial.eval
  %c0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]> : tensor<10xf32>
  %0 = arith.maximumf %x, %c0 : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// CHECK: @test_maximumf_ignore_arg
func.func @test_maximumf_ignore_arg(%x: tensor<10xf32>, %y: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: polynomial.eval
  %0 = arith.maximumf %x, %y : tensor<10xf32>
  return %0 : tensor<10xf32>
}
