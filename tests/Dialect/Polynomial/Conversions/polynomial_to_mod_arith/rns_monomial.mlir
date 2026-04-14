// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

!Z0 = !mod_arith.int<1095233372161 : i64>
!Z1 = !mod_arith.int<1032955396097 : i64>
!rns = !rns.rns<!Z0, !Z1>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**1024>>
!poly = !polynomial.polynomial<ring = #ring>

// CHECK: @test_rns_monomial
// CHECK-SAME: (%[[COEFF:.*]]: !rns.rns<{{.*}}>, %[[DEG:.*]]: index)
func.func @test_rns_monomial(%coeff: !rns, %deg: index) -> !poly {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1024x2xi64>
  // CHECK: %[[ZERO:.*]] = mod_arith.encapsulate %[[CST]] : tensor<1024x2xi64> -> tensor<1024x!rns.rns<{{.*}}>>
  // CHECK: %[[RES:.*]] = tensor.insert %[[COEFF]] into %[[ZERO]][%[[DEG]]]
  %0 = polynomial.monomial %coeff, %deg : (!rns, index) -> !poly
  return %0 : !poly
}
