// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#cycl_2048>
#ring_no = #polynomial.ring<coefficientType = i32, polynomialModulus=#cycl_2048>
#ring_small = #polynomial.ring<coefficientType = i32, coefficientModulus = 17 : i64, polynomialModulus=#cycl_2048>
#ring_prime = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967291 : i64, polynomialModulus=#cycl_2048>

!poly = !polynomial.polynomial<ring=#ring>
!poly_no = !polynomial.polynomial<ring=#ring_no>
!poly_small = !polynomial.polynomial<ring=#ring_small>
!poly_prime = !polynomial.polynomial<ring=#ring_prime>

// CHECK-LABEL: func.func @test_lower_mul_scalar_power_of_two_cmod
// CHECK-SAME: ([[ARG:%.*]]: [[T:.*]])
func.func @test_lower_mul_scalar_power_of_two_cmod(%arg0: !poly) -> !poly {
  // CHECK: [[C2:%.*]] = arith.constant 2 : i32
  // CHECK: [[C2RED:%.+]] = mod_arith.reduce [[C2]] {modulus = 4294967296 : i64} : i32
  %c2 = arith.constant 2 : i32
  // CHECK: [[SPLAT:%.*]] = tensor.splat [[C2RED]] : [[T]]
  // CHECK: mod_arith.mul [[ARG]], [[SPLAT]] {modulus = 4294967296 : i64} : [[T]]
  %8 = polynomial.mul_scalar %arg0, %c2 : !poly, i32
  return %8 : !poly
}

// CHECK-LABEL: func.func @test_lower_mul_scalar_no_cmod
// CHECK-SAME: ([[ARG:%.*]]: [[T:.*]])
func.func @test_lower_mul_scalar_no_cmod(%arg0: !poly_no) -> !poly_no {
  // CHECK: [[C2:%.*]] = arith.constant 2 : i32
  // CHECK: [[C2RED:%.+]] = mod_arith.reduce [[C2]] {modulus = 4294967296 : i34} : i32
  %c2 = arith.constant 2 : i32
  // CHECK: [[SPLAT:%.*]] = tensor.splat [[C2RED]] : [[T]]
  // CHECK: mod_arith.mul [[ARG]], [[SPLAT]] {modulus = 4294967296 : i34} : [[T]]
  %8 = polynomial.mul_scalar %arg0, %c2 : !poly_no, i32
  return %8 : !poly_no
}

// CHECK-LABEL: func.func @test_lower_mul_scalar_small_cmod
// CHECK-SAME: ([[ARG:%.*]]: [[T:.*]])
func.func @test_lower_mul_scalar_small_cmod(%arg0: !poly_small) -> !poly_small {
  // CHECK: [[C2:%.*]] = arith.constant 2 : i32
  // CHECK: [[C2RED:%.+]] = mod_arith.reduce [[C2]] {modulus = 17 : i64} : i32
  %c2 = arith.constant 2 : i32
  // CHECK: [[SPLAT:%.*]] = tensor.splat [[C2RED]] : [[T]]
  // CHECK: mod_arith.mul [[ARG]], [[SPLAT]] {modulus = 17 : i64} : [[T]]
  %8 = polynomial.mul_scalar %arg0, %c2 : !poly_small, i32
  return %8 : !poly_small
}

// CHECK-LABEL: func.func @test_lower_mul_scalar_prime_cmod
// CHECK-SAME: ([[ARG:%.*]]: [[T:.*]])
func.func @test_lower_mul_scalar_prime_cmod(%arg0: !poly_prime) -> !poly_prime {
  // CHECK: [[C2:%.*]] = arith.constant 2 : i32
  // CHECK: [[C2RED:%.+]] = mod_arith.reduce [[C2]] {modulus = 4294967291 : i64} : i32
  %c2 = arith.constant 2 : i32
  // CHECK: [[SPLAT:%.*]] = tensor.splat [[C2RED]] : [[T]]
  // CHECK: mod_arith.mul [[ARG]], [[SPLAT]] {modulus = 4294967291 : i64} : [[T]]
  %8 = polynomial.mul_scalar %arg0, %c2 : !poly_prime, i32
  return %8 : !poly_prime
}
