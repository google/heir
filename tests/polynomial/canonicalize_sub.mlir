// RUN: heir-opt --canonicalize %s | FileCheck %s

#cycl_2048 = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<cmod=4294967296, ideal=#cycl_2048>

// CHECK-LABEL: test_canonicalize_sub_power_of_two_cmod
func.func @test_canonicalize_sub_power_of_two_cmod() -> !polynomial.polynomial<#ring> {
  %poly0 = polynomial.constant <1 + x**2> : !polynomial.polynomial<#ring>
  %poly1 = polynomial.constant <1 + -1x**2> : !polynomial.polynomial<#ring>
  %0 = polynomial.sub(%poly0, %poly1) {ring = #ring} : !polynomial.polynomial<#ring>
  // CHECK: %[[minus_one:.+]] = arith.constant -1 : i32
  // CHECK: %[[p1:.+]] = polynomial.constant
  // CHECK: %[[p2:.+]] = polynomial.constant
  // CHECK: %[[p2neg:.+]] = polynomial.mul_scalar %[[p2]], %[[minus_one]]
  // CHECK: [[ADD:%.+]] = polynomial.add(%[[p1]], %[[p2neg]])
  return %0 : !polynomial.polynomial<#ring>
}
