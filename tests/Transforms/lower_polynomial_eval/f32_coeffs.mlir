// RUN: heir-opt %s --lower-polynomial-eval=method=horner --canonicalize | FileCheck %s

#ring_f32_ = #polynomial.ring<coefficientType = f32>
!poly = !polynomial.polynomial<ring = #ring_f32_>

// CHECK: @test_f32_coeffs(
// CHECK-SAME: [[arg0:%[^ ]*]]: [[ty:f32]]
func.func @test_f32_coeffs(%arg0: f32) -> f32 {
  // CHECK-NEXT: [[c1:%[^ ]*]] = arith.constant 1.5
  // CHECK-NEXT: [[c2:%[^ ]*]] = arith.constant 2.5
  // CHECK-NEXT: [[c3:%[^ ]*]] = arith.constant 3.5
  // CHECK-NEXT: [[c4:%[^ ]*]] = arith.constant 4.9
  // CHECK-NEXT: [[x3_term:%[^ ]*]] = arith.mulf [[arg0]], [[c4]] : [[ty]]
  // CHECK-NEXT: [[sum1:%[^ ]*]] = arith.addf [[x3_term]], [[c3]] : [[ty]]
  // CHECK-NEXT: [[x2_term:%[^ ]*]] = arith.mulf [[sum1]], [[arg0]] : [[ty]]
  // CHECK-NEXT: [[sum2:%[^ ]*]] = arith.addf [[x2_term]], [[c2]] : [[ty]]
  // CHECK-NEXT: [[x1_term:%[^ ]*]] = arith.mulf [[sum2]], [[arg0]] : [[ty]]
  // CHECK-NEXT: [[output:%[^ ]*]] = arith.addf [[x1_term]], [[c1]] : [[ty]]
  // CHECK-NEXT: return [[output]] : [[ty]]
  %0 = polynomial.eval #polynomial<typed_float_polynomial <1.5 + 2.5x + 3.5x**2 + 4.9x**3> : !poly>, %arg0 : f32
  return %0 : f32
}
