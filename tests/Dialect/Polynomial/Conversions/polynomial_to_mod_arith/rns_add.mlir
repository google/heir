// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

!Z0 = !mod_arith.int<1095233372161 : i64>
!Z1 = !mod_arith.int<1032955396097 : i64>
!rns = !rns.rns<!Z0, !Z1>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**1024>>
!poly = !polynomial.polynomial<ring = #ring>

// CHECK: @test_rns_add
// CHECK-SAME: (%[[LHS:.*]]: tensor<1024x!rns.rns<{{.*}}>>, %[[RHS:.*]]: tensor<1024x!rns.rns<{{.*}}>>)
func.func @test_rns_add(%lhs: !poly, %rhs: !poly) -> !poly {
  // CHECK: %[[RES:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : tensor<1024x!rns.rns<{{.*}}>>
  %0 = polynomial.add %lhs, %rhs : !poly
  // CHECK: return %[[RES]]
  return %0 : !poly
}
