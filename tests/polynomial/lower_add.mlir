// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#cycl_2048 = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<cmod=4294967296, ideal=#cycl_2048>
#ring_prime = #polynomial.ring<cmod=4294967291, ideal=#cycl_2048>

func.func @test_lower_add_power_of_two_cmod() -> !polynomial.polynomial<#ring> {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[T:tensor<1024xi32>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
  // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[T]]
  %coeffs2 = arith.constant dense<3> : tensor<1024xi32>
  // CHECK-NOT: polynomial.from_tensor
  %poly0 = polynomial.from_tensor %coeffs1 : tensor<1024xi32> -> !polynomial.polynomial<#ring>
  %poly1 = polynomial.from_tensor %coeffs2 : tensor<1024xi32> -> !polynomial.polynomial<#ring>
  // CHECK-NEXT: [[ADD:%.+]] = arith.addi [[X]], [[Y]]
  %poly2 = polynomial.add(%poly0, %poly1) {ring = #ring} : !polynomial.polynomial<#ring>
  // CHECK: return  [[ADD]] : [[T]]
  return %poly2 : !polynomial.polynomial<#ring>
}

func.func @test_lower_add_prime_cmod() -> !polynomial.polynomial<#ring_prime> {
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi31>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi31>
  // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[TCOEFF]]
  %coeffs2 = arith.constant dense<3> : tensor<1024xi31>
  // CHECK-NOT: polynomial.from_tensor
  // CHECK: [[XEXT:%.+]] = arith.extsi [[X]] : [[TCOEFF]] to [[T:tensor<1024xi32>]]
  // CHECK: [[YEXT:%.+]] = arith.extsi [[Y]] : [[TCOEFF]] to [[T:tensor<1024xi32>]]
  %poly0 = polynomial.from_tensor %coeffs1 : tensor<1024xi31> -> !polynomial.polynomial<#ring_prime>
  %poly1 = polynomial.from_tensor %coeffs2 : tensor<1024xi31> -> !polynomial.polynomial<#ring_prime>

  // CHECK: [[MOD:%.+]] = arith.constant dense<4294967291> : [[T2:tensor<1024xi33>]]
  // CHECK: [[XEXT2:%.+]] = arith.extsi [[XEXT]] : [[T]] to [[T2]]
  // CHECK: [[YEXT2:%.+]] = arith.extsi [[YEXT]] : [[T]] to [[T2]]
  // CHECK: [[ADD_RESULT:%.+]] = arith.addi [[XEXT2]], [[YEXT2]]
  // CHECK: [[REM_RESULT:%.+]] = arith.remsi [[ADD_RESULT]], [[MOD]]
  // CHECK: [[TRUNC_RESULT:%.+]] = arith.trunci [[REM_RESULT]] : [[T2]] to [[T]]
  %poly2 = polynomial.add(%poly0, %poly1) {ring = #ring_prime} : !polynomial.polynomial<#ring_prime>

  // CHECK: return  [[TRUNC_RESULT]] : [[T]]
  return %poly2 : !polynomial.polynomial<#ring_prime>
}
