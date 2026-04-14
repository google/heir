// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

!Z0 = !mod_arith.int<1095233372161 : i64>
!Z1 = !mod_arith.int<1032955396097 : i64>
!rns = !rns.rns<!Z0, !Z1>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**1024>>
!poly = !polynomial.polynomial<ring = #ring>

// CHECK: @test_rns_from_tensor
// CHECK-SAME: (%[[T:.*]]: tensor<2x!rns.rns<{{.*}}>>)
func.func @test_rns_from_tensor(%t: tensor<2x!rns>) -> !poly {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<2xi64>
  // CHECK: %[[PAD_VAL:.*]] = mod_arith.encapsulate %[[CST]] : tensor<2xi64> -> !rns.rns<{{.*}}>
  // CHECK: %[[PAD:.*]] = tensor.pad %[[T]]
  // CHECK:   tensor.yield %[[PAD_VAL]]
  %0 = polynomial.from_tensor %t : tensor<2x!rns> -> !poly
  return %0 : !poly
}
