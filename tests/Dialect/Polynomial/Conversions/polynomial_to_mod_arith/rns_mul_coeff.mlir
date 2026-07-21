// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

!Z17 = !mod_arith.int<17 : i32>
!Z19 = !mod_arith.int<19 : i32>
!rns = !rns.rns<!Z17, !Z19>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**4>>
!poly = !polynomial.polynomial<ring = #ring, form = coeff>

// CHECK: func.func @test_rns_mul_coeff
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x!rns.rns<{{.*}}>>, %[[RHS:.*]]: tensor<4x!rns.rns<{{.*}}>>)
func.func @test_rns_mul_coeff(%lhs: !poly, %rhs: !poly) -> !poly {
  // CHECK: %[[ZERO_STORAGE:.*]] = arith.constant dense<0> : tensor<7x2xi32>
  // CHECK: %[[ZERO:.*]] = mod_arith.encapsulate %[[ZERO_STORAGE]] : tensor<7x2xi32> -> tensor<7x!rns.rns<{{.*}}>>
  // CHECK: %[[PRODUCT:.*]] = linalg.generic
  // CHECK-SAME: ins(%[[LHS]], %[[RHS]]
  // CHECK-SAME: outs(%[[ZERO]]
  // CHECK: mod_arith.mul
  // CHECK: mod_arith.add
  // CHECK: linalg.yield
  // CHECK: %[[RESULT:.*]] = call @__heir_poly_mod_7x_rns_17_i32_19_i32_1_x4(%[[PRODUCT]])
  %0 = polynomial.mul %lhs, %rhs : !poly
  // CHECK: return %[[RESULT]]
  return %0 : !poly
}

// CHECK: func.func private @__heir_poly_mod_7x_rns_17_i32_19_i32_1_x4
// CHECK: %[[INVERSE_STORAGE:.*]] = arith.constant dense<1> : tensor<2xi32>
// CHECK: %[[INVERSE:.*]] = mod_arith.encapsulate %[[INVERSE_STORAGE]] : tensor<2xi32> -> !rns.rns<{{.*}}>
// CHECK: rns.extract_residue {{.*}} {index = 0 : index}
// CHECK: rns.extract_residue {{.*}} {index = 1 : index}
// CHECK: arith.andi
// CHECK: %[[DIVISOR_STORAGE:.*]] = arith.constant dense<{{.*}}> : tensor<7x2xi32>
// CHECK: %[[DIVISOR:.*]] = mod_arith.encapsulate %[[DIVISOR_STORAGE]] : tensor<7x2xi32> -> tensor<7x!rns.rns<{{.*}}>>
// CHECK: mod_arith.mul {{.*}}, %[[INVERSE]]
