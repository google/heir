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
  // CHECK: %[[SLICE:.*]] = rns.extract_slice %[[ARG]] <start = 0, size = 1>
  %slice = polynomial.apply_coefficientwise (%arg0 : !poly) {
  ^body(%coeff: !rns, %degree: index):
    %0 = rns.extract_slice %coeff <start = 0, size = 1> : !rns -> !rns0
    polynomial.yield %0 : !rns0
  } -> !poly0
  // CHECK: return %[[SLICE]]
  return %slice : !poly0
}

!poly_eval = !polynomial.polynomial<ring = #ring, form = eval>
!poly0_eval = !polynomial.polynomial<ring = #ring0, form = eval>

// CHECK: @extract_slice_eval
// CHECK-SAME: (%[[ARG:.*]]: tensor<4096x!{{.*}}>)
func.func @extract_slice_eval(%arg0: !poly_eval) -> !poly0_eval {
  // CHECK-NOT: scf.for
  // CHECK: %[[SLICE:.*]] = rns.extract_slice %[[ARG]] <start = 0, size = 1>
  %slice = polynomial.apply_coefficientwise (%arg0 : !poly_eval) {
  ^body(%coeff: !rns, %degree: index):
    %0 = rns.extract_slice %coeff <start = 0, size = 1> : !rns -> !rns0
    polynomial.yield %0 : !rns0
  } -> !poly0_eval
  // CHECK: return %[[SLICE]]
  return %slice : !poly0_eval
}
