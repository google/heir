// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#ideal = #polynomial.polynomial<1 + x**10>
#ring = #polynomial.ring<cmod=4294967296, ideal=#ideal>

func.func @test_monomial() -> !polynomial.polynomial<#ring> {
  // CHECK: arith.constant dense<[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]> : tensor<10xi32>
  %poly = polynomial.constant <1 + x**2> : !polynomial.polynomial<#ring>
  return %poly : !polynomial.polynomial<#ring>
}
