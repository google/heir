// RUN: heir-opt --canonicalize %s | FileCheck %s

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>

// CHECK-LABEL: test_canonicalize_sub_power_of_two_cmod
func.func @test_canonicalize_sub_power_of_two_cmod() -> !poly.poly<#ring> {
  %poly0 = poly.constant <1 + x**2> : !poly.poly<#ring>
  %poly1 = poly.constant <1 + -1x**2> : !poly.poly<#ring>
  %0 = poly.sub(%poly0, %poly1) {ring = #ring} : !poly.poly<#ring>
  // CHECK: %[[minus_one:.+]] = arith.constant -1 : i32
  // CHECK: %[[p1:.+]] = poly.constant
  // CHECK: %[[p2:.+]] = poly.constant
  // CHECK: %[[p2neg:.+]] = poly.mul_constant %[[p2]], %[[minus_one]]
  // CHECK: [[ADD:%.+]] = poly.add(%[[p1]], %[[p2neg]])
  return %0 : !poly.poly<#ring>
}
