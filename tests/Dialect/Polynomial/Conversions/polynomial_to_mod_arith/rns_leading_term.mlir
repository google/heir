// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

!Z17 = !mod_arith.int<17 : i32>
!Z19 = !mod_arith.int<19 : i32>
!rns = !rns.rns<!Z17, !Z19>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**4>>
!poly = !polynomial.polynomial<ring = #ring, form = coeff>

// CHECK: func.func @test_rns_leading_term
// CHECK-SAME: (%[[COEFFS:.*]]: tensor<4x!rns.rns<{{.*}}>>) -> !rns.rns<{{.*}}>
func.func @test_rns_leading_term(%coeffs: tensor<4x!rns>) -> !rns {
  %poly = polynomial.from_tensor %coeffs : tensor<4x!rns> -> !poly
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK: %[[INIT:.*]] = arith.constant 3 : index
  // CHECK: %[[DEGREE:.*]] = scf.while (%[[INDEX:.*]] = %[[INIT]]) : (index) -> index {
  // CHECK: %[[COEFF:.*]] = tensor.extract %[[COEFFS]][%[[INDEX]]]
  // CHECK: %[[RESIDUE0:.*]] = rns.extract_residue %[[COEFF]] {index = 0 : index}
  // CHECK: %[[LIFTED0:.*]] = mod_arith.lift standard %[[RESIDUE0]]
  // CHECK: %[[ZERO0:.*]] = arith.cmpi eq, %[[LIFTED0]], %[[ZERO]]
  // CHECK: %[[RESIDUE1:.*]] = rns.extract_residue %[[COEFF]] {index = 1 : index}
  // CHECK: %[[LIFTED1:.*]] = mod_arith.lift standard %[[RESIDUE1]]
  // CHECK: %[[ZERO1:.*]] = arith.cmpi eq, %[[LIFTED1]], %[[ZERO]]
  // CHECK: %[[ALL_ZERO:.*]] = arith.andi %[[ZERO0]], %[[ZERO1]]
  // CHECK: scf.condition(%[[ALL_ZERO]]) %[[INDEX]]
  // CHECK: }
  // CHECK: %[[LEADING_COEFF:.*]] = tensor.extract %[[COEFFS]][%[[DEGREE]]]
  %degree, %coefficient = polynomial.leading_term %poly : !poly -> (index, !rns)
  // CHECK: return %[[LEADING_COEFF]]
  return %coefficient : !rns
}
