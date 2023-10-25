// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#cycl_2048 = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<cmod=4294967296, ideal=#cycl_2048>

func.func @test_monomial() -> !polynomial.polynomial<#ring> {
  // CHECK: %[[deg:.*]] = arith.constant 1023
  %deg = arith.constant 1023 : index
  // CHECK: %[[five:.*]] = arith.constant 5
  %five = arith.constant 5 : i32
  // CHECK: %[[container:.*]] = arith.constant dense<0>
  // CHECK: tensor.insert %[[five]] into %[[container]][%[[deg]]]
  %0 = polynomial.monomial %five, %deg : (i32, index) -> !polynomial.polynomial<#ring>
  return %0 : !polynomial.polynomial<#ring>
}
