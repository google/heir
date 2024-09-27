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

// CHECK-LABEL: @test_lower_add_power_of_two_cmod
func.func @test_lower_add_power_of_two_cmod() -> !poly {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[T:tensor<1024xi32>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
  // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[T]]
  %coeffs2 = arith.constant dense<3> : tensor<1024xi32>
  // CHECK-NOT: polynomial.from_tensor
  // CHECK: [[XRED:%.+]] = mod_arith.reduce [[X]] {modulus = 4294967296 : i64} : [[T]]
  %poly0 = polynomial.from_tensor %coeffs1 : tensor<1024xi32> -> !poly
  // CHECK: [[YRED:%.+]] = mod_arith.reduce [[Y]] {modulus = 4294967296 : i64} : [[T]]
  %poly1 = polynomial.from_tensor %coeffs2 : tensor<1024xi32> -> !poly
  // CHECK-NEXT: [[ADD:%.+]] = mod_arith.add [[XRED]], [[YRED]] {modulus = 4294967296 : i64}
  %poly2 = polynomial.add %poly0, %poly1 : !poly
  // CHECK: return [[ADD]] : [[T]]
  return %poly2 : !poly
}

// CHECK-LABEL: @test_lower_add_no_cmod
func.func @test_lower_add_no_cmod() -> !poly_no {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[T:tensor<1024xi32>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
  // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[T]]
  %coeffs2 = arith.constant dense<3> : tensor<1024xi32>
  // CHECK-NOT: polynomial.from_tensor
  // CHECK: [[XRED:%.+]] = mod_arith.reduce [[X]] {modulus = 4294967296 : i34} : [[T]]
  %poly0 = polynomial.from_tensor %coeffs1 : tensor<1024xi32> -> !poly_no
  // CHECK: [[YRED:%.+]] = mod_arith.reduce [[Y]] {modulus = 4294967296 : i34} : [[T]]
  %poly1 = polynomial.from_tensor %coeffs2 : tensor<1024xi32> -> !poly_no
  // CHECK-NEXT: [[ADD:%.+]] = mod_arith.add [[XRED]], [[YRED]] {modulus = 4294967296 : i34}
  %poly2 = polynomial.add %poly0, %poly1 : !poly_no
  // CHECK: return [[ADD]] : [[T]]
  return %poly2 : !poly_no
}

// CHECK-LABEL: @test_lower_add_small_cmod
func.func @test_lower_add_small_cmod() -> !poly_small {
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi31>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi31>
  // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[TCOEFF]]
  %coeffs2 = arith.constant dense<3> : tensor<1024xi31>
  // CHECK-NOT: polynomial.from_tensor
  // CHECK: [[XEXT:%.+]] = arith.extsi [[X]] : [[TCOEFF]] to [[T:.*]]
  // CHECK: [[XRED:%.+]] = mod_arith.reduce [[XEXT]] {modulus = 17 : i64} : [[T]]
  %poly0 = polynomial.from_tensor %coeffs1 : tensor<1024xi31> -> !poly_small
  // CHECK-NOT: polynomial.from_tensor
  // CHECK: [[YEXT:%.+]] = arith.extsi [[Y]] : [[TCOEFF]] to [[T]]
  // CHECK: [[YRED:%.+]] = mod_arith.reduce [[YEXT]] {modulus = 17 : i64} : [[T]]
  %poly1 = polynomial.from_tensor %coeffs2 : tensor<1024xi31> -> !poly_small

  // CHECK: [[ADD_RESULT:%.+]] = mod_arith.add [[XRED]], [[YRED]] {modulus = 17 : i64} : [[T]]
  %poly2 = polynomial.add %poly0, %poly1 : !poly_small

  // CHECK: return [[ADD_RESULT]] : [[T]]
  return %poly2 : !poly_small
}

// CHECK-LABEL: @test_lower_add_prime_cmod
func.func @test_lower_add_prime_cmod() -> !poly_prime {
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi31>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi31>
  // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[TCOEFF]]
  %coeffs2 = arith.constant dense<3> : tensor<1024xi31>
  // CHECK-NOT: polynomial.from_tensor
  // CHECK: [[XEXT:%.+]] = arith.extsi [[X]] : [[TCOEFF]] to [[T:.*]]
  // CHECK: [[XRED:%.+]] = mod_arith.reduce [[XEXT]] {modulus = 4294967291 : i64} : [[T]]
  %poly0 = polynomial.from_tensor %coeffs1 : tensor<1024xi31> -> !poly_prime
  // CHECK-NOT: polynomial.from_tensor
  // CHECK: [[YEXT:%.+]] = arith.extsi [[Y]] : [[TCOEFF]] to [[T]]
  // CHECK: [[YRED:%.+]] = mod_arith.reduce [[YEXT]] {modulus = 4294967291 : i64} : [[T]]
  %poly1 = polynomial.from_tensor %coeffs2 : tensor<1024xi31> -> !poly_prime

  // CHECK: [[ADD_RESULT:%.+]] = mod_arith.add [[XRED]], [[YRED]] {modulus = 4294967291 : i64} : [[T]]
  %poly2 = polynomial.add %poly0, %poly1 : !poly_prime

  // CHECK: return [[ADD_RESULT]] : [[T]]
  return %poly2 : !poly_prime
}
