// RUN: heir-opt %s --lower-polynomial-eval --split-input-file | FileCheck %s

// -----

module attributes {
  backend.lattigo,
  backend.config_override = {has_kernel_chebyshev = true}
} {
  // CHECK: @test_secret_kernel
  func.func @test_secret_kernel(%x: !secret.secret<f64>) -> !secret.secret<f64> {
    // CHECK: secret.generic
    // CHECK: kernel.eval_chebyshev %{{.*}} {coefficients = [1.000000e+00, 2.000000e+00]} : f64 -> f64
    %0 = secret.generic(%x : !secret.secret<f64>) {
    ^body(%x_val: f64):
      %1 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.0, 2.0]> : !polynomial.polynomial<ring=<coefficientType=f64>>, %x_val {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f64
      secret.yield %1 : f64
    } -> (!secret.secret<f64>)
    return %0 : !secret.secret<f64>
  }

  // CHECK: @test_public_kernel
  func.func @test_public_kernel(%x: f64) -> f64 {
    // CHECK-NOT: kernel.eval_chebyshev
    // CHECK: arith.addf
    %0 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.0, 2.0]> : !polynomial.polynomial<ring=<coefficientType=f64>>, %x {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f64
    return %0 : f64
  }

  // CHECK: @eval_chebyshev_scaling
  func.func @eval_chebyshev_scaling(%x: !secret.secret<f64>) -> !secret.secret<f64> {
    // CHECK: secret.generic
    // CHECK-NEXT: ^body(%[[VAL_0:.*]]: f64):
    // CHECK: %[[CST_0:.*]] = arith.constant 5.000000e-01 : f64
    // CHECK: %[[VAL_1:.*]] = arith.mulf %[[VAL_0]], %[[CST_0]] : f64
    // CHECK: %[[CST_1:.*]] = arith.constant -1.000000e+00 : f64
    // CHECK: %[[VAL_2:.*]] = arith.addf %[[VAL_1]], %[[CST_1]] : f64
    // CHECK: kernel.eval_chebyshev %[[VAL_2]] {coefficients = [1.000000e+00, 2.000000e+00]} : f64 -> f64
    %0 = secret.generic(%x : !secret.secret<f64>) {
    ^body(%x_val: f64):
      %1 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.0, 2.0]> : !polynomial.polynomial<ring=<coefficientType=f64>>, %x_val {domain_lower = 0.0 : f64, domain_upper = 4.0 : f64} : f64
      secret.yield %1 : f64
    } -> (!secret.secret<f64>)
    return %0 : !secret.secret<f64>
  }
}

// -----

module attributes {
  backend.lattigo,
  backend.config_override = {has_kernel_chebyshev = false}
} {
  // CHECK: @test_arith
  func.func @test_arith(%x: f64) -> f64 {
    // CHECK-NOT: kernel.eval_chebyshev
    // CHECK: arith.addf
    %0 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.0, 2.0]> : !polynomial.polynomial<ring=<coefficientType=f64>>, %x {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f64
    return %0 : f64
  }
}
