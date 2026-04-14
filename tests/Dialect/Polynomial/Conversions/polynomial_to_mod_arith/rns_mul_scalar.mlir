// RUN: heir-opt --mlir-print-local-scope --polynomial-to-mod-arith %s | FileCheck %s

!Z0 = !mod_arith.int<1095233372161 : i64>
!Z1 = !mod_arith.int<1032955396097 : i64>
!rns = !rns.rns<!Z0, !Z1>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**1024>>
!poly = !polynomial.polynomial<ring = #ring>

// CHECK: func.func @test_rns_mul_scalar
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024x!rns.rns<{{.*}}>>, %[[ARG1:.*]]: !rns.rns<{{.*}}>)
func.func @test_rns_mul_scalar(%poly: !poly, %scalar: !rns) -> !poly {
  // CHECK: %[[EXTRACTED:.*]] = mod_arith.extract %[[ARG1]] : !rns.rns<{{.*}}> -> tensor<2xi64>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1024x2xi64>
  // CHECK: %[[BROADCASTED:.*]] = linalg.broadcast ins(%[[EXTRACTED]] : tensor<2xi64>) outs(%[[EMPTY]] : tensor<1024x2xi64>) dimensions = [0]
  // CHECK: %[[ENCAPSULATED:.*]] = mod_arith.encapsulate %[[BROADCASTED]] : tensor<1024x2xi64> -> tensor<1024x!rns.rns<{{.*}}>>
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[ARG0]], %[[ENCAPSULATED]] : tensor<1024x!rns.rns<{{.*}}>>
  // CHECK: return %[[MUL]] : tensor<1024x!rns.rns<{{.*}}>>
  %0 = polynomial.mul_scalar %poly, %scalar : !poly, !rns
  return %0 : !poly
}
