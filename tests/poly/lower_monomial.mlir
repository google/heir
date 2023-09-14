// RUN: heir-opt --poly-to-standard %s | FileCheck %s

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>

func.func @test_monomial() {
  // CHECK: %[[deg:.*]] = arith.constant 1023
  %deg = arith.constant 1023 : index
  // CHECK: %[[five:.*]] = arith.constant 5
  %five = arith.constant 5 : i32
  // CHECK: %[[container:.*]] = arith.constant dense<0>
  // CHECK: tensor.insert %[[five]] into %[[container]][%[[deg]]]
  %0 = poly.monomial %five, %deg : (i32, index) -> !poly.poly<#ring>
  return
}
