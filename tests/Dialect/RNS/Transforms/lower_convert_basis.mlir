// RUN: heir-opt --rns-lower-convert-basis --mlir-print-local-scope %s | FileCheck %s

!Z3 = !mod_arith.int<3 : i32>
!Z5 = !mod_arith.int<5 : i32>
!Z7 = !mod_arith.int<7 : i32>
!src = !rns.rns<!Z3, !Z5>
!target = !rns.rns<!Z5, !Z7>

// CHECK: convert_basis
func.func @convert_basis(%arg0: !src) -> !target {
  // CHECK: rns.extract_residue
  // CHECK: mod_arith.lift centered
  // CHECK: rns.extract_residue
  // CHECK: mod_arith.encapsulate
  // CHECK: mod_arith.reduce
  // CHECK: mod_arith.sub
  // CHECK: mod_arith.constant 2
  // CHECK: mod_arith.mul
  // CHECK: mod_arith.lift centered
  // CHECK: tensor.from_elements
  // CHECK: rns.extract_residue
  // CHECK: mod_arith.encapsulate
  // CHECK: mod_arith.reduce
  // CHECK: mod_arith.constant 3
  // CHECK: tensor.from_elements
  // CHECK: mod_arith.encapsulate
  // CHECK: mod_arith.reduce
  // CHECK: mod_arith.mac
  // CHECK: rns.pack
  // CHECK-NOT: rns.convert_basis
  %0 = rns.convert_basis %arg0 {targetBasis = !target} : !src -> !target
  return %0 : !target
}
