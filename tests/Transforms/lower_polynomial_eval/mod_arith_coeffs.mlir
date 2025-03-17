// RUN: heir-opt %s --lower-polynomial-eval=method=horner --canonicalize | FileCheck %s

#ring_i32_ = #polynomial.ring<coefficientType = i32>
!poly = !polynomial.polynomial<ring = #ring_i32_>
!Z65537 = !mod_arith.int<65537 : i64>

// CHECK: [[ty:![^ ]*]] = !mod_arith.int<65537 : i64>
// CHECK: @test_mod_arith_coeffs(
// CHECK-SAME: [[arg0:%[^ ]*]]: [[ty]]
func.func @test_mod_arith_coeffs(%arg0: !Z65537) -> !Z65537 {
  // CHECK-NEXT: [[c1:%[^ ]*]] = mod_arith.constant 1 : [[ty]]
  // CHECK-NEXT: [[c2:%[^ ]*]] = mod_arith.constant 2 : [[ty]]
  // CHECK-NEXT: [[c3:%[^ ]*]] = mod_arith.constant 3 : [[ty]]
  // CHECK-NEXT: [[c4:%[^ ]*]] = mod_arith.constant 4 : [[ty]]
  // CHECK-NEXT: [[x3_term:%[^ ]*]] = mod_arith.mul [[arg0]], [[c4]] : [[ty]]
  // CHECK-NEXT: [[sum1:%[^ ]*]] = mod_arith.add [[x3_term]], [[c3]] : [[ty]]
  // CHECK-NEXT: [[x2_term:%[^ ]*]] = mod_arith.mul [[sum1]], [[arg0]] : [[ty]]
  // CHECK-NEXT: [[sum2:%[^ ]*]] = mod_arith.add [[x2_term]], [[c2]] : [[ty]]
  // CHECK-NEXT: [[x1_term:%[^ ]*]] = mod_arith.mul [[sum2]], [[arg0]] : [[ty]]
  // CHECK-NEXT: [[output:%[^ ]*]] = mod_arith.add [[x1_term]], [[c1]] : [[ty]]
  // CHECK-NEXT: return [[output]] : [[ty]]
  %0 = polynomial.eval #polynomial<typed_int_polynomial <1 + 2x + 3x**2 + 4x**3> : !poly>, %arg0 : !Z65537
  return %0 : !Z65537
}
