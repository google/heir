// RUN: heir-opt --canonicalize %s | FileCheck %s

#cycl_2048 = #_polynomial.polynomial<1 + x**1024>
#ring = #_polynomial.ring<cmod=4294967296, ideal=#cycl_2048>

// CHECK-LABEL: test_canonicalize_sub_power_of_two_cmod
func.func @test_canonicalize_sub_power_of_two_cmod() -> !_polynomial.polynomial<#ring> {
  %poly0 = _polynomial.constant <1 + x**2> : !_polynomial.polynomial<#ring>
  %poly1 = _polynomial.constant <1 + -1x**2> : !_polynomial.polynomial<#ring>
  %0 = _polynomial.sub(%poly0, %poly1) {ring = #ring} : !_polynomial.polynomial<#ring>
  // CHECK: %[[minus_one:.+]] = arith.constant -1 : i32
  // CHECK: %[[p1:.+]] = _polynomial.constant
  // CHECK: %[[p2:.+]] = _polynomial.constant
  // CHECK: %[[p2neg:.+]] = _polynomial.mul_scalar %[[p2]], %[[minus_one]]
  // CHECK: [[ADD:%.+]] = _polynomial.add(%[[p1]], %[[p2neg]])
  return %0 : !_polynomial.polynomial<#ring>
}
