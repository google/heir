// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

!Z0 = !mod_arith.int<270337 : i64>
!Z1 = !mod_arith.int<188417 : i64>
!Z2 = !mod_arith.int<286721 : i64>
!rns = !rns.rns<!Z0, !Z1, !Z2>
!rns0 = !rns.rns<!Z0>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**4096>>
#ring0 = #polynomial.ring<coefficientType = !rns0, polynomialModulus = <1 + x**4096>>
!poly = !polynomial.polynomial<ring = #ring>
!poly0 = !polynomial.polynomial<ring = #ring0>

// CHECK: @extract_slice
// CHECK-SAME: (%[[ARG:.*]]: tensor<4096x!{{.*}}>)
func.func @extract_slice(%arg0: !poly) -> !poly0 {
  // CHECK: %[[SLICE:.*]] = rns.extract_slice %[[ARG]] {size = 1 : index, start = 0 : index}
  %slice = polynomial.extract_slice %arg0 {start = 0 : index, size = 1 : index} : !poly -> !poly0
  // CHECK: return %[[SLICE]]
  return %slice : !poly0
}
