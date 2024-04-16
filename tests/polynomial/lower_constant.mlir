// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#ideal = #_polynomial.polynomial<1 + x**10>
#ring = #_polynomial.ring<cmod=4294967296, ideal=#ideal>

func.func @test_monomial() -> !_polynomial.polynomial<#ring> {
  // CHECK: arith.constant dense<[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]> : tensor<10xi32>
  %poly = _polynomial.constant <1 + x**2> : !_polynomial.polynomial<#ring>
  return %poly : !_polynomial.polynomial<#ring>
}
