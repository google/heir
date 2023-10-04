// RUN: heir-opt --poly-to-standard %s | FileCheck %s

#ideal = #poly.polynomial<1 + x**10>
#ring = #poly.ring<cmod=4294967296, ideal=#ideal>

func.func @test_monomial() {
  // CHECK: arith.constant dense<[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]> : tensor<10xi32>
  %poly = poly.constant <1 + x**2> : !poly.poly<#ring>
  return
}
