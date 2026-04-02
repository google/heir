// RUN: heir-opt --rns-lower-convert-basis --mlir-print-local-scope %s | FileCheck %s

!Z3 = !mod_arith.int<3 : i32>
!Z5 = !mod_arith.int<5 : i32>
!Z7 = !mod_arith.int<7 : i32>
!src = !rns.rns<!Z3, !Z5>
!target = !rns.rns<!Z5, !Z7>

// CHECK: convert_basis
func.func @convert_basis(%arg0: !src) -> !target {
  // CHECK: rns.extract_residue
  // CHECK-NEXT: mod_arith.extract
  // CHECK-NEXT: rns.extract_residue
  // CHECK-NEXT: mod_arith.encapsulate
  // CHECK-NEXT: mod_arith.reduce
  // CHECK-NEXT: mod_arith.sub
  // CHECK-NEXT: mod_arith.constant 2
  // CHECK-NEXT: mod_arith.mul
  // CHECK-NEXT: mod_arith.extract
  // CHECK-NEXT: rns.extract_residue
  // CHECK-NEXT: mod_arith.encapsulate
  // CHECK-NEXT: mod_arith.reduce
  // CHECK-NEXT: mod_arith.constant 3
  // CHECK-NEXT: mod_arith.encapsulate
  // CHECK-NEXT: mod_arith.reduce
  // CHECK-NEXT: mod_arith.mac
  // CHECK-NEXT: rns.pack
  // CHECK-NOT: rns.convert_basis
  %0 = rns.convert_basis %arg0 {targetBasis = !target} : !src -> !target
  return %0 : !target
}
